from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from experiments.esc50_pooling_tradeoff.data import build_dataset
from experiments.esc50_pooling_tradeoff.gradcam_figure import (
    build_model,
    find_checkpoint,
    infer_num_classes_from_csv,
    read_rows,
)
from experiments.esc50_pooling_tradeoff.pooling import AdaptiveSSRP_T, SSRP_T
from experiments.esc50_pooling_tradeoff.train_eval_cv import resolve_device


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create paper-style SSRP region visualization figure.")
    p.add_argument("--run-dir", required=True, help="Run directory containing fold subdirectories")
    p.add_argument("--fold", required=True, help="Fold name, e.g. fold1 or fold_4")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--sample-indices",
        required=True,
        help="Comma-separated sample indices to place as figure columns, e.g. 0,10,50",
    )
    p.add_argument("--device", default="cuda", help="cuda, cpu, or auto")
    p.add_argument("--fig-dpi", type=int, default=200)
    return p.parse_args()


def compute_window_means(feat: torch.Tensor, window: int) -> torch.Tensor:
    # feat: (C, F, T)
    C, Freq, T = feat.shape
    if T < window:
        return feat.unsqueeze(-1).mean(dim=-1, keepdim=True)
    if window == 1:
        return feat
    x = feat.contiguous().view(C * Freq, 1, T)
    wmean = F.avg_pool1d(x, kernel_size=window, stride=1)
    return wmean.view(C, Freq, wmean.size(-1))


def compute_selection_mask(feat: torch.Tensor, window: int, topk: int) -> torch.Tensor:
    # feat: (C, F, T), returns normalized map (F, T)
    C, Freq, T = feat.shape
    wmean = compute_window_means(feat, window)
    Tw = wmean.size(-1)
    k_eff = min(int(topk), int(Tw))
    topk_idx = torch.topk(wmean, k=k_eff, dim=-1).indices  # (C,F,K)
    mask = torch.zeros((C, Freq, T), dtype=torch.float32, device=feat.device)
    w_eff = 1 if T < window else int(window)
    for offset in range(w_eff):
        idx = (topk_idx + offset).clamp_max(T - 1)
        mask.scatter_add_(-1, idx, torch.ones_like(idx, dtype=torch.float32))
    mask = mask.mean(dim=0)  # (F,T)
    mask = mask / mask.max().clamp_min(1e-12)
    return mask


def compute_adaptive_alpha(pool: AdaptiveSSRP_T, feat_b: torch.Tensor) -> torch.Tensor:
    # feat_b: (1,C,F,T)
    g_in = pool._compute_gate_input(feat_b)
    logits = pool.gate(g_in)
    temp = pool._current_temperature()
    alpha = torch.softmax(logits / temp, dim=-1)
    if pool.alpha_floor > 0.0:
        alpha = (1.0 - pool.alpha_floor) * alpha + pool.alpha_floor / pool.num_k
    return alpha.squeeze(0)


def prepare_sample_maps(model, x: torch.Tensor, y_true: int) -> Dict:
    with torch.no_grad():
        feat = model.backbone(x)  # (1,C,F,T)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(probs.argmax().item())
        pred_prob = float(probs[pred_idx].item())

    feat0 = feat.squeeze(0)  # (C,F,T)
    input_map = x.squeeze(0).squeeze(0).detach().cpu()
    activation_map = feat0.abs().mean(dim=0)
    activation_map = activation_map / activation_map.max().clamp_min(1e-12)

    if isinstance(model.pool, SSRP_T):
        ks = [int(model.pool.K)]
        masks = [compute_selection_mask(feat0, int(model.pool.W), int(model.pool.K))]
        weighted = masks[0]
        mode = "ssrp_t"
        alpha = None
    elif isinstance(model.pool, AdaptiveSSRP_T):
        ks = list(model.pool.Ks)
        alpha = compute_adaptive_alpha(model.pool, feat)
        masks = [compute_selection_mask(feat0, int(model.pool.W), int(k)) for k in ks]
        weighted = sum(float(alpha[i].item()) * masks[i] for i in range(len(ks)))
        weighted = weighted / weighted.max().clamp_min(1e-12)
        mode = "adaptive_ssrp_t"
    else:
        raise ValueError(f"Unsupported pooling for paper-style SSRP figure: {type(model.pool).__name__}")

    selected_activation = activation_map * weighted
    selected_activation = selected_activation / selected_activation.max().clamp_min(1e-12)

    return {
        "input_map": input_map.cpu().numpy(),
        "activation_map": activation_map.cpu().numpy(),
        "selected_activation": selected_activation.cpu().numpy(),
        "masks": [m.cpu().numpy() for m in masks],
        "weighted_mask": weighted.cpu().numpy(),
        "ks": ks,
        "alpha": alpha.detach().cpu().numpy().tolist() if alpha is not None else None,
        "pred_class": pred_idx,
        "pred_prob": pred_prob,
        "true_class": int(y_true),
        "mode": mode,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    fold_dir = run_dir / args.fold
    split_csv = fold_dir / "normalized_csv" / f"{args.split}.csv"
    ckpt_path = find_checkpoint(fold_dir)
    state = torch.load(ckpt_path, map_location="cpu")

    cfg = state["config"]
    data_cfg = dict(cfg["data"])
    model_cfg = dict(cfg["model"])
    device = torch.device(resolve_device(args.device))

    rows = read_rows(split_csv)
    sample_indices = parse_int_list(args.sample_indices)
    if not sample_indices:
        raise ValueError("No sample indices provided.")

    data_cfg["root"] = str(Path.cwd())
    dataset = build_dataset(data_cfg, model_cfg, str(split_csv), training=False)
    num_classes = int(state.get("num_classes") or data_cfg.get("num_classes") or infer_num_classes_from_csv(split_csv))
    model = build_model(cfg, num_classes=num_classes).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()

    samples: List[Dict] = []
    for sample_index in sample_indices:
        if not (0 <= sample_index < len(rows)):
            raise IndexError(f"sample-index {sample_index} out of range for {split_csv} ({len(rows)} rows)")
        x, y = dataset[sample_index]
        x = x.unsqueeze(0).to(device)
        sample = prepare_sample_maps(model, x, int(y.item()))
        sample["sample_index"] = int(sample_index)
        sample["sample_row"] = rows[sample_index]
        samples.append(sample)

    first = samples[0]
    row_defs: List[Tuple[str, str]] = [("input", "Input")]
    for k in first["ks"]:
        row_defs.append((f"mask_k{k}", f"Selected Regions (K={k})"))
    if first["mode"] == "adaptive_ssrp_t":
        row_defs.append(("weighted", "Adaptive Weighted Regions"))
    row_defs.append(("gap_like", "GAP Activation Map"))
    row_defs.append(("ssrp_like", "SSRP Activation Map"))

    n_rows = len(row_defs)
    n_cols = len(samples)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 2.2 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for col, sample in enumerate(samples):
        col_title = sample["sample_row"].get("class_name", f"idx{sample['sample_index']}")
        axes[0, col].set_title(col_title)

        for row_idx, (key, label) in enumerate(row_defs):
            ax = axes[row_idx, col]
            if key == "input":
                arr = sample["input_map"]
                ax.imshow(arr, origin="lower", aspect="auto", cmap="magma")
            elif key.startswith("mask_k"):
                k = int(key.replace("mask_k", ""))
                k_idx = sample["ks"].index(k)
                ax.imshow(sample["masks"][k_idx], origin="lower", aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
            elif key == "weighted":
                ax.imshow(sample["weighted_mask"], origin="lower", aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
            elif key == "gap_like":
                ax.imshow(sample["activation_map"], origin="lower", aspect="auto", cmap="jet", vmin=0.0, vmax=1.0)
            elif key == "ssrp_like":
                ax.imshow(sample["selected_activation"], origin="lower", aspect="auto", cmap="jet", vmin=0.0, vmax=1.0)
            else:
                raise KeyError(key)

            if col == 0:
                ax.set_ylabel(label)
            ax.set_xlabel("Temporal Frame Index")
            if key == "input":
                ax.set_ylabel("Mel Frequency Index" if col == 0 else "")

    fig.suptitle(f"{run_dir.name} | {args.fold} | {args.split}", fontsize=11)

    out_dir = fold_dir / "paper_regions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_samples_{'_'.join(str(i) for i in sample_indices)}"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    json_path = out_dir / f"{stem}.json"

    fig.savefig(png_path, dpi=int(args.fig_dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "run_dir": str(run_dir),
        "fold": args.fold,
        "checkpoint_path": str(ckpt_path),
        "split": args.split,
        "sample_indices": sample_indices,
        "row_defs": row_defs,
        "samples": [
            {
                "sample_index": int(sample["sample_index"]),
                "sample_row": sample["sample_row"],
                "ks": sample["ks"],
                "alpha": sample["alpha"],
                "pred_class": int(sample["pred_class"]),
                "pred_prob": float(sample["pred_prob"]),
                "true_class": int(sample["true_class"]),
                "mode": sample["mode"],
            }
            for sample in samples
        ],
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("[Done]")
    print(f"figure_png : {png_path}")
    print(f"figure_pdf : {pdf_path}")
    print(f"meta_json  : {json_path}")


if __name__ == "__main__":
    main()
