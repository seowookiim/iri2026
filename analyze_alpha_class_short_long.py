from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .data import build_dataset, build_loader
from .models import PoolingAudioClassifier
from .train_eval_cv import resolve_device, temporal_peak_mass_score, unpack_logits_and_alpha


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze adaptive alpha by class for short-term vs long-term samples."
    )
    p.add_argument("--run-dir", required=True, help="Run directory that contains fold*/best*.pt")
    p.add_argument("--device", default="cuda", help="cuda, cpu, or auto")
    p.add_argument("--show-pbar", action="store_true", help="Show tqdm while iterating test sets")
    return p.parse_args()


def find_checkpoint(fold_dir: Path) -> Path:
    cands = sorted(fold_dir.glob("best*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No checkpoint found in {fold_dir}")
    return cands[0]


def build_model(cfg: Dict, num_classes: int) -> PoolingAudioClassifier:
    model_cfg = cfg["model"]
    return PoolingAudioClassifier(
        backbone=model_cfg.get("backbone", "lightweight_cnn"),
        num_classes=num_classes,
        pooling=str(model_cfg.get("pooling", "adaptive_ssrp_t")).lower(),
        ssrp_w=int(model_cfg.get("ssrp_w", 4)),
        ssrp_k=int(model_cfg.get("ssrp_k", 12)),
        ssrp_b_k=int(model_cfg.get("ssrp_b_k", 12)),
        adaptive_ks=tuple(model_cfg.get("adaptive_ks", [4, 8, 12])),
        adaptive_gate_hidden=int(model_cfg.get("adaptive_gate_hidden", 128)),
        adaptive_gate_input=str(model_cfg.get("adaptive_gate_input", "meanstd")),
        adaptive_gate_norm=bool(model_cfg.get("adaptive_gate_norm", True)),
        adaptive_gate_dropout=float(model_cfg.get("adaptive_gate_dropout", 0.0)),
        adaptive_temperature=float(model_cfg.get("adaptive_temperature", 1.5)),
        adaptive_learnable_temperature=bool(model_cfg.get("adaptive_learnable_temperature", False)),
        adaptive_alpha_floor=float(model_cfg.get("adaptive_alpha_floor", 0.0)),
        adaptive_use_branch_calibration=bool(model_cfg.get("adaptive_use_branch_calibration", False)),
        adaptive_return_mode="details",
        head_hidden=int(model_cfg.get("head_hidden", 128)),
        dropout=float(model_cfg.get("dropout", 0.5)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
        return_alpha=True,
    )


def load_class_names(test_csv: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with test_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["class_index"])] = str(row.get("class_name", row["class_index"]))
    return out


@torch.inference_mode()
def collect_fold_stats(
    fold_dir: Path,
    device: torch.device,
) -> Dict:
    ckpt_path = find_checkpoint(fold_dir)
    state = torch.load(ckpt_path, map_location="cpu")
    cfg = state["config"]
    num_classes = int(state.get("num_classes") or cfg["data"]["num_classes"])

    test_csv = fold_dir / "normalized_csv" / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing normalized test.csv: {test_csv}")

    data_cfg = dict(cfg["data"])
    data_cfg["root"] = str(Path.cwd())

    test_set = build_dataset(data_cfg, cfg["model"], str(test_csv), training=False)
    test_loader = build_loader(
        test_set,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        num_workers=int(cfg["data"].get("num_workers", 0)),
        training=False,
    )

    model = build_model(cfg, num_classes=num_classes).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()

    class_names = load_class_names(test_csv)
    alpha_batches: List[torch.Tensor] = []
    y_batches: List[torch.Tensor] = []
    short_batches: List[torch.Tensor] = []

    for batch in test_loader:
        if batch is None:
            continue
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        output = model(x)
        _, alpha = unpack_logits_and_alpha(output)
        if alpha is None:
            raise RuntimeError(f"No alpha returned for fold {fold_dir.name}. Is pooling adaptive?")

        alpha_batches.append(alpha.detach().cpu().to(torch.float64))
        y_batches.append(y.detach().cpu())
        short_batches.append(temporal_peak_mass_score(x).detach().cpu().to(torch.float64))

    alpha_all = torch.cat(alpha_batches, dim=0)
    y_all = torch.cat(y_batches, dim=0)
    shortness_all = torch.cat(short_batches, dim=0)

    threshold = float(torch.quantile(shortness_all, q=0.5).item())
    short_mask = shortness_all > threshold
    long_mask = ~short_mask
    if int(short_mask.sum().item()) == 0 or int(long_mask.sum().item()) == 0:
        short_mask = shortness_all >= threshold
        long_mask = shortness_all < threshold

    by_class: Dict[int, Dict[str, Dict[str, object]]] = {}
    for cls_id in sorted(class_names):
        cls_mask = y_all == cls_id
        short_cls = cls_mask & short_mask
        long_cls = cls_mask & long_mask

        def pack(mask: torch.Tensor) -> Dict[str, object]:
            n = int(mask.sum().item())
            if n <= 0:
                return {"count": 0, "alpha_mean": None, "shortness_mean": None}
            a = alpha_all[mask]
            s = shortness_all[mask]
            return {
                "count": n,
                "alpha_mean": a.mean(dim=0).tolist(),
                "shortness_mean": float(s.mean().item()),
            }

        by_class[cls_id] = {
            "class_name": class_names[cls_id],
            "short_term": pack(short_cls),
            "long_term": pack(long_cls),
        }

    return {
        "fold": fold_dir.name,
        "checkpoint_path": str(ckpt_path),
        "threshold": threshold,
        "ks": list(cfg["model"].get("adaptive_ks", [4, 8, 12])),
        "class_stats": by_class,
    }


def aggregate_folds(fold_payloads: List[Dict]) -> Dict:
    ks = fold_payloads[0]["ks"]
    acc: Dict[int, Dict[str, object]] = {}
    for payload in fold_payloads:
        for cls_id, cls_stats in payload["class_stats"].items():
            cls_bucket = acc.setdefault(
                cls_id,
                {
                    "class_name": cls_stats["class_name"],
                    "short_term": {"count": 0, "alpha_sum": np.zeros(len(ks), dtype=np.float64), "shortness_sum": 0.0},
                    "long_term": {"count": 0, "alpha_sum": np.zeros(len(ks), dtype=np.float64), "shortness_sum": 0.0},
                },
            )
            for split_name in ("short_term", "long_term"):
                src = cls_stats[split_name]
                if not src["count"]:
                    continue
                cls_bucket[split_name]["count"] += int(src["count"])
                cls_bucket[split_name]["alpha_sum"] += np.asarray(src["alpha_mean"], dtype=np.float64) * int(src["count"])
                cls_bucket[split_name]["shortness_sum"] += float(src["shortness_mean"]) * int(src["count"])

    out_classes = {}
    for cls_id in sorted(acc):
        row = {"class_name": acc[cls_id]["class_name"]}
        for split_name in ("short_term", "long_term"):
            bucket = acc[cls_id][split_name]
            n = int(bucket["count"])
            if n <= 0:
                row[split_name] = {"count": 0, "alpha_mean": None, "shortness_mean": None}
            else:
                row[split_name] = {
                    "count": n,
                    "alpha_mean": (bucket["alpha_sum"] / n).tolist(),
                    "shortness_mean": float(bucket["shortness_sum"] / n),
                }
        out_classes[str(cls_id)] = row

    return {"ks": ks, "class_stats": out_classes}


def write_csv(path: Path, aggregate: Dict) -> None:
    ks = aggregate["ks"]
    headers = [
        "class_index",
        "class_name",
        "short_count",
        "long_count",
        "shortness_mean_short",
        "shortness_mean_long",
    ]
    headers += [f"short_alpha_k{k}" for k in ks]
    headers += [f"long_alpha_k{k}" for k in ks]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for cls_id, row in aggregate["class_stats"].items():
            short = row["short_term"]
            long = row["long_term"]
            writer.writerow(
                [
                    cls_id,
                    row["class_name"],
                    short["count"],
                    long["count"],
                    short["shortness_mean"],
                    long["shortness_mean"],
                    *([] if short["alpha_mean"] is None else short["alpha_mean"]),
                    *([] if long["alpha_mean"] is None else long["alpha_mean"]),
                ]
            )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)

    fold_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("fold")])
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found in {run_dir}")

    fold_payloads = [collect_fold_stats(fd, device=device) for fd in fold_dirs]
    aggregate = aggregate_folds(fold_payloads)

    payload = {
        "run_dir": str(run_dir),
        "metric": "top10_energy_mass",
        "split": "median_by_fold",
        "folds": fold_payloads,
        "aggregate": aggregate,
    }

    json_path = run_dir / "alpha_class_short_long.json"
    csv_path = run_dir / "alpha_class_short_long.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, aggregate)

    print("[Done]")
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")


if __name__ == "__main__":
    main()
