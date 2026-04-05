from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.esc50_pooling_tradeoff.data import build_dataset
from experiments.esc50_pooling_tradeoff.gradcam_figure import (
    GradCAM,
    build_model,
    choose_default_target_layer,
    disable_inplace_relu,
    find_checkpoint,
    get_module_by_name,
    infer_num_classes_from_csv,
    read_rows,
)
from experiments.esc50_pooling_tradeoff.train_eval_cv import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare GAP / SSRP-T / AdaptiveSSRP-T Grad-CAM on the same sample."
    )
    p.add_argument("--gap-run-dir", required=True)
    p.add_argument("--ssrp-run-dir", required=True)
    p.add_argument("--adaptive-run-dir", required=True)
    p.add_argument("--fold", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument(
        "--sample-indices",
        default=None,
        help="Comma-separated sample indices for multi-column figure, e.g. 0,10,50,100",
    )
    p.add_argument("--auto-select", action="store_true", help="Automatically select representative samples.")
    p.add_argument("--num-samples", type=int, default=4, help="Number of columns when --auto-select is used.")
    p.add_argument(
        "--selection-mode",
        default="mixed",
        choices=["representative", "advantage", "mixed"],
        help="Auto-selection strategy.",
    )
    p.add_argument(
        "--target-mode",
        default="true",
        choices=["true", "pred"],
        help="Class used to compute CAM. 'true' uses the sample label, 'pred' uses each model prediction.",
    )
    p.add_argument("--target-layer", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha", type=float, default=0.45)
    return p.parse_args()


def parse_int_list(value: Optional[str], fallback: int) -> List[int]:
    if value is None or not str(value).strip():
        return [int(fallback)]
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def _sample_activity_stats(x_np: np.ndarray) -> Dict[str, float]:
    energy_t = np.abs(x_np).mean(axis=0)
    peak = float(np.max(energy_t)) if energy_t.size > 0 else 0.0
    if peak <= 1e-12:
        return {"active_fraction": 0.0, "active_span_fraction": 0.0}

    mask = energy_t > (0.12 * peak)
    active_fraction = float(mask.mean())
    if not np.any(mask):
        return {"active_fraction": active_fraction, "active_span_fraction": 0.0}

    idx = np.where(mask)[0]
    span = int(idx[-1] - idx[0] + 1)
    active_span_fraction = float(span / max(1, energy_t.shape[0]))
    return {
        "active_fraction": active_fraction,
        "active_span_fraction": active_span_fraction,
    }


def _load_sample(run_dir: Path, fold: str, split: str, sample_index: int):
    fold_dir = run_dir / fold
    split_csv = fold_dir / "normalized_csv" / f"{split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing normalized split CSV: {split_csv}")
    rows = read_rows(split_csv)
    if not (0 <= sample_index < len(rows)):
        raise IndexError(f"sample-index {sample_index} out of range for {split_csv} ({len(rows)} rows)")

    ckpt_path = find_checkpoint(fold_dir)
    state = torch.load(ckpt_path, map_location="cpu")
    cfg = state["config"]
    data_cfg = dict(cfg["data"])
    model_cfg = dict(cfg["model"])
    data_cfg["root"] = str(Path.cwd())
    dataset = build_dataset(data_cfg, model_cfg, str(split_csv), training=False)
    x, y = dataset[sample_index]
    return {
        "fold_dir": fold_dir,
        "split_csv": split_csv,
        "rows": rows,
        "row": rows[sample_index],
        "ckpt_path": ckpt_path,
        "state": state,
        "cfg": cfg,
        "x": x,
        "y_true": int(y.item()),
    }


def _build_run_context(run_dir: Path, fold: str, split: str, device: torch.device) -> Dict:
    fold_dir = run_dir / fold
    split_csv = fold_dir / "normalized_csv" / f"{split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing normalized split CSV: {split_csv}")
    rows = read_rows(split_csv)
    ckpt_path = find_checkpoint(fold_dir)
    state = torch.load(ckpt_path, map_location="cpu")
    cfg = state["config"]
    data_cfg = dict(cfg["data"])
    model_cfg = dict(cfg["model"])
    data_cfg["root"] = str(Path.cwd())
    dataset = build_dataset(data_cfg, model_cfg, str(split_csv), training=False)
    num_classes = int(
        state.get("num_classes")
        or cfg["data"].get("num_classes")
        or infer_num_classes_from_csv(split_csv)
    )
    pooling_name = str(cfg["model"].get("pooling", "gap")).lower()
    model = build_model(cfg, num_classes=num_classes)
    if pooling_name in {"adaptive_ssrp_t", "adaptive_ssrp"}:
        model.return_alpha = True
    model = model.to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    disable_inplace_relu(model)
    return {
        "run_dir": run_dir,
        "fold_dir": fold_dir,
        "split_csv": split_csv,
        "rows": rows,
        "state": state,
        "cfg": cfg,
        "dataset": dataset,
        "model": model,
        "pooling_name": pooling_name,
    }


@torch.inference_mode()
def _predict_from_context(ctx: Dict, sample_index: int, device: torch.device) -> Dict:
    x, y = ctx["dataset"][sample_index]
    x = x.unsqueeze(0).to(device)
    out = ctx["model"](x)
    alpha = None
    if isinstance(out, tuple):
        logits = out[0]
        aux = out[1]
        if isinstance(aux, dict):
            alpha = aux.get("alpha")
        else:
            alpha = aux
    else:
        logits = out
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
    pred_idx = int(probs.argmax().item())
    pred_prob = float(probs[pred_idx].item())
    alpha_list = None
    if alpha is not None:
        alpha_cpu = alpha.detach().cpu()
        if alpha_cpu.ndim == 2 and alpha_cpu.size(0) > 0:
            alpha_list = alpha_cpu[0].tolist()
    x_np = x.detach().cpu().squeeze(0).numpy()
    activity = _sample_activity_stats(x_np)
    return {
        "true_class": int(y.item()),
        "pred_class": pred_idx,
        "pred_prob": pred_prob,
        "alpha": alpha_list,
        "row": ctx["rows"][sample_index],
        "x": x_np,
        **activity,
    }


def _score_candidate(gap_pred: Dict, ssrp_pred: Dict, adapt_pred: Dict, mode: str) -> Optional[float]:
    true_cls = adapt_pred["true_class"]
    gap_ok = gap_pred["pred_class"] == true_cls
    ssrp_ok = ssrp_pred["pred_class"] == true_cls
    adapt_ok = adapt_pred["pred_class"] == true_cls
    active_frac = float(adapt_pred.get("active_fraction", 0.0))
    active_span = float(adapt_pred.get("active_span_fraction", 0.0))

    # Exclude heavily padded or nearly full-band uniform samples for the paper figure.
    if active_frac < 0.08:
        return None
    if active_span < 0.15 or active_span > 0.92:
        return None

    activity_bonus = 1.0 - abs(active_span - 0.45)

    if mode == "representative":
        if not (gap_ok and ssrp_ok and adapt_ok):
            return None
        return (
            adapt_pred["pred_prob"]
            - 0.5 * (gap_pred["pred_prob"] + ssrp_pred["pred_prob"])
            + 0.5 * activity_bonus
        )

    if mode == "advantage":
        if not adapt_ok:
            return None
        if gap_ok and ssrp_ok:
            return None
        bonus = (0 if gap_ok else 1) + (0 if ssrp_ok else 1)
        return (
            10.0 * bonus
            + adapt_pred["pred_prob"]
            - max(gap_pred["pred_prob"], ssrp_pred["pred_prob"])
            + 0.5 * activity_bonus
        )

    # mixed
    if gap_ok and ssrp_ok and adapt_ok:
        return (
            2.0
            + adapt_pred["pred_prob"]
            - 0.5 * (gap_pred["pred_prob"] + ssrp_pred["pred_prob"])
            + 0.5 * activity_bonus
        )
    if adapt_ok and (not gap_ok or not ssrp_ok):
        bonus = (0 if gap_ok else 1) + (0 if ssrp_ok else 1)
        return (
            10.0 * bonus
            + adapt_pred["pred_prob"]
            - max(gap_pred["pred_prob"], ssrp_pred["pred_prob"])
            + 0.5 * activity_bonus
        )
    return None


def _auto_select_sample_indices(
    gap_ctx: Dict,
    ssrp_ctx: Dict,
    adapt_ctx: Dict,
    device: torch.device,
    mode: str,
    num_samples: int,
) -> List[int]:
    n = len(adapt_ctx["rows"])
    candidates: List[Dict] = []
    for idx in range(n):
        gap_pred = _predict_from_context(gap_ctx, idx, device)
        ssrp_pred = _predict_from_context(ssrp_ctx, idx, device)
        adapt_pred = _predict_from_context(adapt_ctx, idx, device)
        score = _score_candidate(gap_pred, ssrp_pred, adapt_pred, mode)
        if score is None:
            continue
        candidates.append(
            {
                "idx": idx,
                "score": float(score),
                "class_name": str(adapt_pred["row"].get("class_name", f"class {adapt_pred['true_class']}")),
                "active_fraction": float(adapt_pred.get("active_fraction", 0.0)),
                "active_span_fraction": float(adapt_pred.get("active_span_fraction", 0.0)),
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected: List[int] = []
    seen_classes = set()
    for item in candidates:
        if item["class_name"] in seen_classes:
            continue
        selected.append(int(item["idx"]))
        seen_classes.add(item["class_name"])
        if len(selected) >= int(num_samples):
            break

    if len(selected) < int(num_samples):
        for idx in range(n):
            cls = str(adapt_ctx["rows"][idx].get("class_name", f"class_{idx}"))
            if cls in seen_classes:
                continue
            selected.append(int(idx))
            seen_classes.add(cls)
            if len(selected) >= int(num_samples):
                break
    return selected


def _compute_cam(
    run_dir: Path,
    fold: str,
    split: str,
    sample_index: int,
    device: torch.device,
    target_mode: str,
    target_layer_name: Optional[str],
) -> Dict:
    pack = _load_sample(run_dir, fold, split, sample_index)
    state = pack["state"]
    cfg = pack["cfg"]

    num_classes = int(
        state.get("num_classes")
        or cfg["data"].get("num_classes")
        or infer_num_classes_from_csv(pack["split_csv"])
    )
    pooling_name = str(cfg["model"].get("pooling", "gap")).lower()
    model = build_model(cfg, num_classes=num_classes)
    if pooling_name in {"adaptive_ssrp_t", "adaptive_ssrp"}:
        model.return_alpha = True
    model = model.to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    disable_inplace_relu(model)

    x = pack["x"].unsqueeze(0).to(device)
    y_true = pack["y_true"]

    with torch.enable_grad():
        out0 = model(x)
        alpha = None
        if isinstance(out0, tuple):
            logits0 = out0[0]
            aux = out0[1]
            if isinstance(aux, dict):
                alpha = aux.get("alpha")
            else:
                alpha = aux
        else:
            logits0 = out0

        pred_idx = int(logits0.argmax(dim=1).item())
        class_idx = y_true if target_mode == "true" else pred_idx

        layer_name = target_layer_name or choose_default_target_layer(model)
        layer = get_module_by_name(model, layer_name)
        cam_engine = GradCAM(model, layer)
        try:
            logits, cam = cam_engine.compute(x, class_idx)
        finally:
            cam_engine.remove()

    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
    pred_prob = float(probs[pred_idx].item())

    alpha_list = None
    if alpha is not None:
        alpha_cpu = alpha.detach().cpu()
        if alpha_cpu.ndim == 2 and alpha_cpu.size(0) > 0:
            alpha_list = alpha_cpu[0].tolist()

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "pooling": pooling_name,
        "fold_dir": str(pack["fold_dir"]),
        "ckpt_path": str(pack["ckpt_path"]),
        "split_csv": str(pack["split_csv"]),
        "row": pack["row"],
        "x": pack["x"].detach().cpu().squeeze(0).numpy(),
        "cam": cam.detach().cpu().numpy(),
        "true_class": y_true,
        "pred_class": pred_idx,
        "pred_prob": pred_prob,
        "target_class": class_idx,
        "target_layer": layer_name,
        "alpha": alpha_list,
    }


def _overlay(ax, mel: np.ndarray, cam: np.ndarray, alpha: float) -> None:
    ax.imshow(mel, origin="lower", aspect="auto", cmap="gray")
    ax.imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=alpha)
    ax.set_xlabel("")
    ax.set_ylabel("")


def main() -> None:
    args = parse_args()
    device = torch.device(resolve_device(args.device))
    gap_ctx = _build_run_context(Path(args.gap_run_dir), args.fold, args.split, device)
    ssrp_ctx = _build_run_context(Path(args.ssrp_run_dir), args.fold, args.split, device)
    adapt_ctx = _build_run_context(Path(args.adaptive_run_dir), args.fold, args.split, device)

    if args.auto_select:
        sample_indices = _auto_select_sample_indices(
            gap_ctx=gap_ctx,
            ssrp_ctx=ssrp_ctx,
            adapt_ctx=adapt_ctx,
            device=device,
            mode=args.selection_mode,
            num_samples=args.num_samples,
        )
    else:
        sample_indices = parse_int_list(args.sample_indices, args.sample_index)

    ssrp_cfg = ssrp_ctx["cfg"]["model"]
    out_dir = Path(args.adaptive_run_dir) / args.fold / "compare_gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_idxs{'-'.join(f'{i:04d}' for i in sample_indices)}_gap_ssrp_adapt"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    json_path = out_dir / f"{stem}.json"

    ncols = len(sample_indices)
    fig, axes = plt.subplots(4, ncols, figsize=(4.8 * ncols, 10.8), constrained_layout=True)
    if ncols == 1:
        axes = np.asarray(axes).reshape(4, 1)

    payload_samples = []
    adaptive_cfg = adapt_ctx["cfg"]["model"]
    alpha_labels = [f"K={int(k)}" for k in adaptive_cfg.get("adaptive_ks", [4, 8, 12])]

    for col, sample_index in enumerate(sample_indices):
        gap = _compute_cam(
            run_dir=Path(args.gap_run_dir),
            fold=args.fold,
            split=args.split,
            sample_index=sample_index,
            device=device,
            target_mode=args.target_mode,
            target_layer_name=args.target_layer,
        )
        ssrp = _compute_cam(
            run_dir=Path(args.ssrp_run_dir),
            fold=args.fold,
            split=args.split,
            sample_index=sample_index,
            device=device,
            target_mode=args.target_mode,
            target_layer_name=args.target_layer,
        )
        adaptive = _compute_cam(
            run_dir=Path(args.adaptive_run_dir),
            fold=args.fold,
            split=args.split,
            sample_index=sample_index,
            device=device,
            target_mode=args.target_mode,
            target_layer_name=args.target_layer,
        )

        mel = adaptive["x"]
        class_name = str(adaptive["row"].get("class_name", f"class {adaptive['true_class']}"))
        axes[0, col].set_title(class_name, fontsize=11)
        axes[0, col].imshow(mel, origin="lower", aspect="auto", cmap="magma")
        axes[0, col].set_xlabel("")
        axes[0, col].set_ylabel("")

        _overlay(axes[1, col], mel, gap["cam"], float(args.alpha))
        _overlay(axes[2, col], mel, ssrp["cam"], float(args.alpha))
        _overlay(axes[3, col], mel, adaptive["cam"], float(args.alpha))

        payload_samples.append(
            {
                "sample_index": int(sample_index),
                "class_name": class_name,
                "gap": {k: v for k, v in gap.items() if k not in {"x", "cam"}},
                "ssrp_t": {k: v for k, v in ssrp.items() if k not in {"x", "cam"}},
                "adaptive_ssrp_t": {k: v for k, v in adaptive.items() if k not in {"x", "cam"}},
            }
        )

    row_labels = [
        "Input Log-Mel",
        "GAP",
        f"SSRP-T (W={int(ssrp_cfg.get('ssrp_w', 4))}, K={int(ssrp_cfg.get('ssrp_k', 12))})",
        "AdaptiveSSRP-T",
    ]
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=11)
    for row_idx in range(4):
        for col in range(1, ncols):
            axes[row_idx, col].set_ylabel("")
    for row_idx in range(4):
        for col in range(ncols):
            if row_idx < 3:
                axes[row_idx, col].set_xticklabels([])
            if col > 0:
                axes[row_idx, col].set_yticklabels([])
    for col in range(ncols):
        axes[3, col].set_xlabel("Time", fontsize=10)
    for row_idx in range(4):
        axes[row_idx, 0].set_yticks([0, 10, 20, 30])

    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    payload = {
        "fold": args.fold,
        "split": args.split,
        "sample_indices": [int(v) for v in sample_indices],
        "target_mode": args.target_mode,
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "samples": payload_samples,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("[Done]")
    print(f"sample_indices: {sample_indices}")
    print(f"figure_png : {png_path}")
    print(f"figure_pdf : {pdf_path}")
    print(f"meta_json  : {json_path}")
    print(
        "runs       : "
        f"GAP={Path(args.gap_run_dir).name}, "
        f"SSRP={Path(args.ssrp_run_dir).name}, "
        f"ADAPT={Path(args.adaptive_run_dir).name}"
    )


if __name__ == "__main__":
    main()
