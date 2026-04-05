from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from experiments.esc50_pooling_tradeoff.data import build_dataset
from experiments.esc50_pooling_tradeoff.models import PoolingAudioClassifier
from experiments.esc50_pooling_tradeoff.train_eval_cv import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Grad-CAM figures from a saved run/fold checkpoint.")
    p.add_argument("--run-dir", required=True, help="Run directory that contains fold subdirectories")
    p.add_argument("--fold", required=True, help="Fold directory name, e.g. fold1 or fold_4")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--sample-index", type=int, default=0, help="Index within selected split CSV")
    p.add_argument("--target-class", type=int, default=None, help="Explicit target class for CAM; default uses predicted class")
    p.add_argument(
        "--target-layer",
        default=None,
        help="Module path for CAM target, e.g. backbone.features.10. Default picks the final conv block automatically.",
    )
    p.add_argument("--device", default="cuda", help="cuda, cpu, or auto")
    p.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for CAM heatmap")
    return p.parse_args()


def find_checkpoint(fold_dir: Path) -> Path:
    cands = sorted(fold_dir.glob("best*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No best checkpoint found in: {fold_dir}")
    return cands[0]


def infer_num_classes_from_csv(csv_path: Path) -> int:
    max_idx = -1
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            max_idx = max(max_idx, int(row["class_index"]))
    return max_idx + 1


def build_model(cfg: Dict, num_classes: int) -> PoolingAudioClassifier:
    model_cfg = cfg["model"]
    return PoolingAudioClassifier(
        backbone=model_cfg.get("backbone", "lightweight_cnn"),
        num_classes=num_classes,
        pooling=str(model_cfg.get("pooling", "ssrp_t")).lower(),
        ssrp_w=int(model_cfg.get("ssrp_w", 4)),
        ssrp_k=int(model_cfg.get("ssrp_k", 12)),
        ssrp_b_k=int(model_cfg.get("ssrp_b_k", 12)),
        asp_attention_hidden=int(model_cfg.get("asp_attention_hidden", 128)),
        adaptive_ks=tuple(model_cfg.get("adaptive_ks", [4, 8, 12])),
        adaptive_gate_hidden=int(model_cfg.get("adaptive_gate_hidden", 128)),
        adaptive_gate_input=str(model_cfg.get("adaptive_gate_input", "meanstd")),
        adaptive_gate_norm=bool(model_cfg.get("adaptive_gate_norm", True)),
        adaptive_gate_dropout=float(model_cfg.get("adaptive_gate_dropout", 0.0)),
        adaptive_temperature=float(model_cfg.get("adaptive_temperature", 1.5)),
        adaptive_learnable_temperature=bool(model_cfg.get("adaptive_learnable_temperature", False)),
        adaptive_alpha_floor=float(model_cfg.get("adaptive_alpha_floor", 0.0)),
        adaptive_use_branch_calibration=bool(model_cfg.get("adaptive_use_branch_calibration", False)),
        adaptive_return_mode=str(model_cfg.get("adaptive_return_mode", "z")),
        head_hidden=int(model_cfg.get("head_hidden", 128)),
        dropout=float(model_cfg.get("dropout", 0.5)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
        return_alpha=False,
    )


def disable_inplace_relu(module: torch.nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.ReLU):
            submodule.inplace = False


def choose_default_target_layer(model: PoolingAudioClassifier) -> str:
    if hasattr(model.backbone, "features"):
        modules = list(model.backbone.features.named_children())
        for name, module in reversed(modules):
            if isinstance(module, torch.nn.ReLU):
                return f"backbone.features.{name}"
        for name, module in reversed(modules):
            if isinstance(module, torch.nn.Conv2d):
                return f"backbone.features.{name}"
    if hasattr(model.backbone, "tfeb"):
        modules = list(model.backbone.tfeb.named_children())
        for name, module in reversed(modules):
            if isinstance(module, torch.nn.ReLU):
                return f"backbone.tfeb.{name}"
        for name, module in reversed(modules):
            if isinstance(module, torch.nn.Conv2d):
                return f"backbone.tfeb.{name}"
    raise RuntimeError("Could not infer a default CAM target layer from this backbone.")


def get_module_by_name(root: torch.nn.Module, name: str) -> torch.nn.Module:
    module = root
    for part in name.split("."):
        if not part:
            continue
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()

    def compute(self, x: torch.Tensor, class_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations/gradients for Grad-CAM.")

        acts = self.activations
        grads = self.gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-12)
        return logits.detach(), cam.detach()


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    fold_dir = run_dir / args.fold
    ckpt_path = find_checkpoint(fold_dir)
    state = torch.load(ckpt_path, map_location="cpu")

    cfg = state["config"]
    data_cfg = dict(cfg["data"])
    model_cfg = dict(cfg["model"])
    device = torch.device(resolve_device(args.device))

    split_csv = fold_dir / "normalized_csv" / f"{args.split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing normalized split CSV: {split_csv}")

    rows = read_rows(split_csv)
    if not (0 <= args.sample_index < len(rows)):
        raise IndexError(f"sample-index {args.sample_index} out of range for {split_csv} ({len(rows)} rows)")

    data_cfg["root"] = str(Path.cwd())
    dataset = build_dataset(data_cfg, model_cfg, str(split_csv), training=False)
    x, y = dataset[args.sample_index]
    x = x.unsqueeze(0).to(device)
    y_true = int(y.item())

    num_classes = int(state.get("num_classes") or data_cfg.get("num_classes") or infer_num_classes_from_csv(split_csv))
    model = build_model(cfg, num_classes=num_classes).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    disable_inplace_relu(model)

    target_layer_name = args.target_layer or choose_default_target_layer(model)
    target_layer = get_module_by_name(model, target_layer_name)
    cam_engine = GradCAM(model, target_layer)
    try:
        with torch.enable_grad():
            logits0 = model(x)
            if isinstance(logits0, tuple):
                logits0 = logits0[0]
            pred_idx = int(logits0.argmax(dim=1).item())
            class_idx = int(args.target_class) if args.target_class is not None else pred_idx
            logits, cam = cam_engine.compute(x, class_idx)
    finally:
        cam_engine.remove()

    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    pred_prob = float(probs[pred_idx].item())

    mel = x.detach().cpu().squeeze(0).squeeze(0)
    cam_np = cam.cpu().numpy()
    mel_np = mel.numpy()

    out_dir = fold_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_idx{args.sample_index:04d}_cls{class_idx}"
    png_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"
    npy_path = out_dir / f"{stem}_cam.npy"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].imshow(mel_np, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("Input log-mel")
    axes[1].imshow(cam_np, origin="lower", aspect="auto", cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(mel_np, origin="lower", aspect="auto", cmap="gray")
    axes[2].imshow(cam_np, origin="lower", aspect="auto", cmap="jet", alpha=float(args.alpha))
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel bin")
    fig.suptitle(
        f"{run_dir.name} | {args.fold} | pred={pred_idx} ({pred_prob:.3f}) | true={y_true} | target={class_idx}",
        fontsize=10,
    )
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    np.save(npy_path, cam_np)
    payload = {
        "run_dir": str(run_dir),
        "fold": args.fold,
        "checkpoint_path": str(ckpt_path),
        "split": args.split,
        "sample_index": int(args.sample_index),
        "sample_row": rows[args.sample_index],
        "true_class": y_true,
        "pred_class": pred_idx,
        "pred_prob": pred_prob,
        "target_class": class_idx,
        "target_layer": target_layer_name,
        "png_path": str(png_path),
        "cam_npy_path": str(npy_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("[Done]")
    print(f"figure_png : {png_path}")
    print(f"meta_json  : {json_path}")
    print(f"cam_npy    : {npy_path}")
    print(f"target_layer: {target_layer_name}")
    print(f"pred_class={pred_idx} prob={pred_prob:.4f} true_class={y_true} target_class={class_idx}")


if __name__ == "__main__":
    main()
