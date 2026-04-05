from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

from .data import build_dataset, build_loader, discover_folds, infer_input_shape
from .metrics import classification_metrics, count_parameters, measure_latency_ms, try_compute_flops
from .models import PoolingAudioClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name and device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_int_list_arg(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    items = [v.strip() for v in str(value).split(",") if v.strip()]
    if not items:
        return None
    return [int(v) for v in items]


def move_to_device(batch, device: torch.device):
    if batch is None:
        return None
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def unpack_logits_and_alpha(output):
    if isinstance(output, tuple):
        aux = output[1]
        if isinstance(aux, dict):
            return output[0], aux.get("alpha")
        return output[0], aux
    return output, None


def temporal_peak_mass_score(x: torch.Tensor, frac: float = 0.1) -> torch.Tensor:
    """
    Compute per-sample temporal concentration score from input tensor.
    Higher value means more short-term/peaky temporal energy.
    """
    if x.dim() < 2:
        raise ValueError(f"Expected x.dim() >= 2, got shape={tuple(x.shape)}")

    xt = x.detach().float()
    reduce_dims = tuple(range(1, xt.dim() - 1))
    if reduce_dims:
        temporal_energy = xt.abs().mean(dim=reduce_dims)  # (B, T)
    else:
        temporal_energy = xt.abs()  # (B, T)

    T = int(temporal_energy.size(-1))
    k = max(1, int(round(T * frac)))
    mass = temporal_energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p_t = temporal_energy / mass
    return torch.topk(p_t, k=k, dim=-1).values.sum(dim=-1)  # (B,)


def alpha_entropy(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = alpha.clamp_min(eps)
    ent = -(a * a.log()).sum(dim=-1)
    return ent.mean()


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[idx]
    return mixed, y, y[idx], float(lam)


def infer_num_classes_from_fold(fold: Dict[str, str]) -> int:
    max_idx = -1
    for key in ("train_csv", "val_csv", "test_csv"):
        if key not in fold or not fold.get(key):
            continue
        with Path(fold[key]).open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                max_idx = max(max_idx, int(r["class_index"]))
    return max_idx + 1


def _sample_rel_paths(csv_path: str, max_rows: int = 32) -> List[str]:
    rels: List[str] = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = str(row.get("path", "")).strip()
            if p:
                rels.append(p)
            if len(rels) >= max_rows:
                break
    return rels


def _root_hit_ratio(root: Path, rel_paths: List[str]) -> float:
    if not rel_paths:
        return 0.0
    hit = 0
    for rel in rel_paths:
        if (root / rel).exists():
            hit += 1
    return hit / len(rel_paths)


def resolve_data_root(config_root: str, fold: Dict[str, str]) -> str:
    configured = Path(config_root)
    rels = _sample_rel_paths(fold["train_csv"])
    if not rels:
        return str(configured)

    candidates = [configured]
    if configured.name.lower() == "audio_by_class":
        candidates.append(configured.parent / "audio")
    if configured.name.lower() == "audio":
        candidates.append(configured.parent / "audio_by_class")
    candidates.extend(
        [
            configured / "audio",
            configured / "audio_by_class",
            configured.parent / "audio",
            configured.parent / "audio_by_class",
            Path.cwd() / "datasets" / "ESC-50" / "audio",
            Path.cwd() / "datasets" / "ESC-50" / "audio_by_class",
            Path.cwd() / "ESC-50-master" / "audio",
        ]
    )

    uniq: List[Path] = []
    seen = set()
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists():
            uniq.append(c)

    best_root = configured
    best_ratio = _root_hit_ratio(configured, rels)
    for c in uniq:
        ratio = _root_hit_ratio(c, rels)
        if ratio > best_ratio:
            best_root = c
            best_ratio = ratio

    if best_ratio <= 0.0:
        probe = rels[0]
        print(
            "[root-resolve] warning: no direct root match for fold paths. "
            f"configured_root={configured}, sample_rel_path={probe}. "
            "Will continue with CSV-level path normalization."
        )
        return str(configured)

    if best_root != configured:
        print(
            f"[root-resolve] configured={configured} -> selected={best_root} "
            f"(hit_ratio={best_ratio:.2f})"
        )
    return str(best_root)


def _candidate_audio_roots(config_root: str) -> List[Path]:
    root = Path(config_root)
    parent = root.parent
    cands = [
        root,
        root / "audio",
        root / "audio_by_class",
        parent / "audio",
        parent / "audio_by_class",
        Path.cwd() / "datasets" / "ESC-50" / "audio",
        Path.cwd() / "datasets" / "ESC-50" / "audio_by_class",
        Path.cwd() / "ESC-50-master" / "audio",
    ]
    uniq: List[Path] = []
    seen = set()
    for c in cands:
        k = str(c)
        if k in seen:
            continue
        seen.add(k)
        if c.exists():
            uniq.append(c)
    return uniq


def _build_basename_index(roots: List[Path]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for r in roots:
        for p in r.rglob("*.wav"):
            name = p.name
            # keep first hit; ESC-50 filenames are unique enough in practice
            if name not in idx:
                idx[name] = str(p.resolve())
    return idx


def normalize_csv_paths(
    csv_in: str,
    csv_out: str,
    config_root: str,
) -> str:
    roots = _candidate_audio_roots(config_root)
    if not roots:
        raise FileNotFoundError(f"No candidate audio roots found near: {config_root}")
    basename_index = _build_basename_index(roots)

    rows = []
    with Path(csv_in).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rel = str(row.get("path", "")).strip()
            if not rel:
                raise ValueError(f"Empty path in CSV: {csv_in}")

            resolved = None
            p = Path(rel)
            if p.is_absolute() and p.exists():
                resolved = p
            else:
                for r in roots:
                    cand = (r / rel)
                    if cand.exists():
                        resolved = cand
                        break
                if resolved is None:
                    base = p.name
                    abs_hit = basename_index.get(base)
                    if abs_hit:
                        resolved = Path(abs_hit)

            if resolved is None:
                raise FileNotFoundError(
                    f"Could not resolve audio path from CSV row: path='{rel}', csv='{csv_in}'"
                )

            row["path"] = str(resolved.resolve())
            rows.append(row)

    out = Path(csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(out)


def merge_csv_unique_by_path(csv_a: str, csv_b: str, csv_out: str) -> str:
    rows = []
    seen = set()
    fieldnames = None

    for src in (csv_a, csv_b):
        with Path(src).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                p = str(row.get("path", "")).strip()
                if not p or p in seen:
                    continue
                seen.add(p)
                rows.append(row)

    out = Path(csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(out)


def run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    num_classes: int,
    mixup_alpha: float = 0.0,
    entropy_lambda: float = 0.0,
    show_pbar: bool = True,
    progress_desc: str = "Train",
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0
    total_correct = 0
    ys_true: List[torch.Tensor] = []
    ys_pred: List[torch.Tensor] = []
    t_values = set()
    total_alpha_entropy = 0.0
    total_alpha_count = 0

    iterable = tqdm(loader, desc=progress_desc, leave=False, disable=not show_pbar, dynamic_ncols=True)
    for batch in iterable:
        batch = move_to_device(batch, device)
        if batch is None:
            continue
        x, y = batch

        with torch.set_grad_enabled(is_train):
            if is_train and mixup_alpha > 0.0:
                x_mix, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
                output = model(x_mix)
            else:
                output = model(x)
            logits, alpha = unpack_logits_and_alpha(output)
            if is_train and mixup_alpha > 0.0:
                loss_task = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                loss_task = criterion(logits, y)
            loss = loss_task

            if is_train and entropy_lambda > 0.0 and alpha is not None:
                ent = alpha_entropy(alpha)
                loss = loss_task - float(entropy_lambda) * ent
                total_alpha_entropy += float(ent.detach().item()) * y.size(0)
                total_alpha_count += int(y.size(0))

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        if getattr(model, "last_feature_shape", None) is not None:
            t_values.add(int(model.last_feature_shape[-1]))

        preds = logits.argmax(dim=1)
        ys_true.append(y.detach().cpu())
        ys_pred.append(preds.detach().cpu())
        bsz = y.size(0)
        total_loss += float(loss.item()) * bsz
        total_count += bsz
        total_correct += int((preds == y).sum().item())
        if show_pbar and total_count > 0:
            iterable.set_postfix(
                loss=f"{(total_loss / total_count):.4f}",
                acc=f"{(total_correct / total_count):.3f}",
            )

    if total_count == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "feature_time_lengths": [],
            "alpha_entropy": None,
        }

    y_true = torch.cat(ys_true, dim=0)
    y_pred = torch.cat(ys_pred, dim=0)
    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    metrics["loss"] = total_loss / total_count
    metrics["feature_time_lengths"] = sorted(t_values)
    metrics["alpha_entropy"] = (
        (total_alpha_entropy / total_alpha_count) if total_alpha_count > 0 else None
    )
    return metrics


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
    collect_alpha_stats: bool = False,
    show_pbar: bool = False,
    progress_desc: str = "Eval",
):
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    ys_true: List[torch.Tensor] = []
    ys_pred: List[torch.Tensor] = []
    t_values = set()

    alpha_sum = None
    alpha_count = 0
    alpha_hist = None
    class_alpha_sum: Dict[int, torch.Tensor] = {}
    class_alpha_count: Dict[int, int] = {}
    alpha_samples: List[torch.Tensor] = []
    shortness_scores: List[torch.Tensor] = []

    iterable = tqdm(loader, desc=progress_desc, leave=False, disable=not show_pbar, dynamic_ncols=True)
    for batch in iterable:
        batch = move_to_device(batch, device)
        if batch is None:
            continue
        x, y = batch
        output = model(x)
        logits, alpha = unpack_logits_and_alpha(output)
        loss = criterion(logits, y)

        if getattr(model, "last_feature_shape", None) is not None:
            t_values.add(int(model.last_feature_shape[-1]))

        preds = logits.argmax(dim=1)
        ys_true.append(y.detach().cpu())
        ys_pred.append(preds.detach().cpu())
        total_correct += int((preds == y).sum().item())

        if collect_alpha_stats and alpha is not None:
            alpha_cpu = alpha.detach().cpu().to(torch.float64)
            y_cpu = y.detach().cpu()

            if alpha_sum is None:
                alpha_sum = alpha_cpu.sum(dim=0)
                alpha_hist = torch.zeros(alpha_cpu.size(1), dtype=torch.float64)
            else:
                alpha_sum += alpha_cpu.sum(dim=0)

            alpha_count += int(alpha_cpu.size(0))
            alpha_hist += torch.bincount(alpha_cpu.argmax(dim=1), minlength=alpha_cpu.size(1)).to(torch.float64)
            alpha_samples.append(alpha_cpu)
            shortness_scores.append(temporal_peak_mass_score(x).detach().cpu().to(torch.float64))

            for cls_id, a in zip(y_cpu.tolist(), alpha_cpu):
                if cls_id not in class_alpha_sum:
                    class_alpha_sum[cls_id] = a.clone()
                    class_alpha_count[cls_id] = 1
                else:
                    class_alpha_sum[cls_id] += a
                    class_alpha_count[cls_id] += 1

        bsz = y.size(0)
        total_loss += float(loss.item()) * bsz
        total_count += bsz
        if show_pbar and total_count > 0:
            iterable.set_postfix(
                loss=f"{(total_loss / total_count):.4f}",
                acc=f"{(total_correct / total_count):.3f}",
            )

    if total_count == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "feature_time_lengths": [],
            "alpha_stats": None,
        }

    y_true = torch.cat(ys_true, dim=0)
    y_pred = torch.cat(ys_pred, dim=0)
    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    metrics["loss"] = total_loss / total_count
    metrics["feature_time_lengths"] = sorted(t_values)

    alpha_stats = None
    if collect_alpha_stats and alpha_sum is not None and alpha_count > 0:
        class_means = {}
        for cls_id in sorted(class_alpha_sum.keys()):
            class_means[str(cls_id)] = (
                class_alpha_sum[cls_id] / max(class_alpha_count.get(cls_id, 1), 1)
            ).tolist()

        alpha_mean = alpha_sum / alpha_count
        dominant_idx = int(alpha_hist.argmax().item()) if alpha_hist is not None else None
        dominant_ratio = (
            float(alpha_hist.max().item() / alpha_count)
            if alpha_hist is not None and alpha_count > 0
            else None
        )

        alpha_all = torch.cat(alpha_samples, dim=0) if alpha_samples else None
        shortness_all = torch.cat(shortness_scores, dim=0) if shortness_scores else None

        entropy_mean = None
        entropy_norm_mean = None
        effective_k_mean = None
        collapse_flags = None
        temporal_conditioning = None

        if alpha_all is not None and alpha_all.numel() > 0:
            m = int(alpha_all.size(1))
            a_safe = alpha_all.clamp_min(1e-12)
            ent = -(a_safe * a_safe.log()).sum(dim=1)  # (N,)
            entropy_mean = float(ent.mean().item())
            if m > 1:
                ent_norm = ent / np.log(float(m))
                entropy_norm_mean = float(ent_norm.mean().item())
            else:
                ent_norm = torch.zeros_like(ent)
                entropy_norm_mean = 0.0
            effective_k_mean = float(torch.exp(ent).mean().item())
            collapse_flags = {
                "argmax_dominant_gt_90pct": bool(
                    (dominant_ratio is not None) and (dominant_ratio > 0.90)
                ),
                "entropy_norm_lt_20pct": bool(entropy_norm_mean < 0.20),
            }

            if shortness_all is not None and int(shortness_all.numel()) == int(alpha_all.size(0)):
                th = float(torch.quantile(shortness_all, q=0.5).item())
                short_mask = shortness_all > th
                long_mask = ~short_mask
                if int(short_mask.sum().item()) == 0 or int(long_mask.sum().item()) == 0:
                    short_mask = shortness_all >= th
                    long_mask = shortness_all < th

                def _cond_pack(mask: torch.Tensor) -> Dict[str, object]:
                    n = int(mask.sum().item())
                    if n <= 0:
                        return {
                            "count": 0,
                            "alpha_mean": None,
                            "alpha_argmax_hist_ratio": None,
                            "shortness_mean": None,
                        }
                    a_sel = alpha_all[mask]
                    h_sel = torch.bincount(a_sel.argmax(dim=1), minlength=a_sel.size(1)).to(torch.float64) / n
                    return {
                        "count": n,
                        "alpha_mean": a_sel.mean(dim=0).tolist(),
                        "alpha_argmax_hist_ratio": h_sel.tolist(),
                        "shortness_mean": float(shortness_all[mask].mean().item()),
                    }

                temporal_conditioning = {
                    "metric": "top10_energy_mass",
                    "split": "median",
                    "threshold": th,
                    "short_term": _cond_pack(short_mask),
                    "long_term": _cond_pack(long_mask),
                }

        alpha_stats = {
            "alpha_mean": alpha_mean.tolist(),
            "alpha_argmax_hist": alpha_hist.tolist() if alpha_hist is not None else None,
            "alpha_argmax_dominant_index": dominant_idx,
            "alpha_argmax_dominant_ratio": dominant_ratio,
            "alpha_entropy_mean": entropy_mean,
            "alpha_entropy_norm_mean": entropy_norm_mean,
            "alpha_effective_k_mean": effective_k_mean,
            "alpha_collapse_flags": collapse_flags,
            "temporal_conditioning": temporal_conditioning,
            "alpha_mean_by_class": class_means,
            "num_samples": alpha_count,
        }

    metrics["alpha_stats"] = alpha_stats
    return metrics


def merge_config(base_cfg: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("output", {})

    if args.root:
        cfg["data"]["root"] = args.root
    if args.folds_dir:
        cfg["data"]["folds_dir"] = args.folds_dir
    if args.out_dir:
        cfg["output"]["out_dir"] = args.out_dir
    if args.run_name:
        cfg["output"]["run_name"] = args.run_name
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.pooling:
        cfg["model"]["pooling"] = args.pooling
    if args.input_representation:
        cfg["model"]["input_representation"] = args.input_representation
    if args.freeze_backbone:
        cfg["model"]["freeze_backbone"] = True
    if args.ssrp_w is not None:
        cfg["model"]["ssrp_w"] = int(args.ssrp_w)
    if args.ssrp_k is not None:
        cfg["model"]["ssrp_k"] = int(args.ssrp_k)
    if args.ssrp_b_k is not None:
        cfg["model"]["ssrp_b_k"] = int(args.ssrp_b_k)
    if args.adaptive_ks is not None:
        parsed = parse_int_list_arg(args.adaptive_ks)
        if parsed:
            cfg["model"]["adaptive_ks"] = parsed
    if args.adaptive_gate_hidden is not None:
        cfg["model"]["adaptive_gate_hidden"] = int(args.adaptive_gate_hidden)
    if args.best_metric:
        cfg["train"]["best_metric"] = args.best_metric
    if getattr(args, "early_stopping_patience", None) is not None:
        cfg["train"]["early_stopping_patience"] = int(args.early_stopping_patience)
    if getattr(args, "early_stopping_min_delta", None) is not None:
        cfg["train"]["early_stopping_min_delta"] = float(args.early_stopping_min_delta)
    if args.cv_protocol:
        cfg["train"]["cv_protocol"] = args.cv_protocol
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)
    return cfg


def _is_better(score: float, best: Optional[float], metric_name: str) -> bool:
    if best is None:
        return True
    if metric_name == "val_loss":
        return score < best
    return score > best


def _selection_score(metric_name: str, val_metrics: Dict[str, float]) -> float:
    if metric_name == "val_accuracy":
        return float(val_metrics["accuracy"])
    if metric_name == "val_macro_f1":
        return float(val_metrics["macro_f1"])
    if metric_name == "val_loss":
        return float(val_metrics["loss"])
    raise ValueError(f"Unknown best metric: {metric_name}")


def _scheduler_score(
    metric_name: str,
    val_metrics: Optional[Dict[str, float]],
    train_metrics: Dict[str, float],
) -> Optional[float]:
    if metric_name == "val_accuracy":
        return float(val_metrics["accuracy"]) if val_metrics is not None else None
    if metric_name == "val_macro_f1":
        return float(val_metrics["macro_f1"]) if val_metrics is not None else None
    if metric_name == "val_loss":
        return float(val_metrics["loss"]) if val_metrics is not None else None
    if metric_name == "train_accuracy":
        return float(train_metrics["accuracy"])
    if metric_name == "train_macro_f1":
        return float(train_metrics["macro_f1"])
    if metric_name == "train_loss":
        return float(train_metrics["loss"])
    raise ValueError(f"Unknown scheduler metric: {metric_name}")


def train_one_fold(
    fold: Dict[str, str],
    cfg: dict,
    device: torch.device,
    run_out_dir: Path,
) -> Dict[str, float]:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg_local = copy.deepcopy(data_cfg)
    data_cfg_local["root"] = resolve_data_root(str(data_cfg["root"]), fold)

    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    best_metric = str(train_cfg.get("best_metric", "val_accuracy"))
    scheduler_name = str(train_cfg.get("scheduler", "reduce_on_plateau")).lower()
    scheduler_monitor = str(train_cfg.get("scheduler_monitor", best_metric))
    scheduler_factor = float(train_cfg.get("scheduler_factor", 0.5))
    scheduler_patience = int(train_cfg.get("scheduler_patience", 20))
    scheduler_threshold = float(train_cfg.get("scheduler_threshold", 1e-4))
    scheduler_cooldown = int(train_cfg.get("scheduler_cooldown", 0))
    scheduler_min_lr = float(train_cfg.get("scheduler_min_lr", 1e-5))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    checkpoint_timestamp = bool(train_cfg.get("checkpoint_timestamp", True))
    artifact_timestamp = bool(train_cfg.get("artifact_timestamp", True))
    cv_protocol = str(train_cfg.get("cv_protocol", "with_val")).lower()
    log_alpha_stats = bool(train_cfg.get("log_alpha_stats", True))
    use_tqdm = bool(train_cfg.get("use_tqdm", True))
    epoch_log = bool(train_cfg.get("epoch_log", True))
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))
    entropy_lambda = float(train_cfg.get("adaptive_entropy_lambda", train_cfg.get("entropy_lambda", 0.0)))
    num_workers = int(data_cfg_local.get("num_workers", 4))

    fold_out = run_out_dir / fold["name"]
    fold_out.mkdir(parents=True, exist_ok=True)
    normalized_dir = fold_out / "normalized_csv"
    train_csv_norm = normalize_csv_paths(
        csv_in=fold["train_csv"],
        csv_out=str(normalized_dir / "train.csv"),
        config_root=str(data_cfg_local["root"]),
    )
    val_csv_norm = None
    if fold.get("val_csv"):
        val_csv_norm = normalize_csv_paths(
            csv_in=fold["val_csv"],
            csv_out=str(normalized_dir / "val.csv"),
            config_root=str(data_cfg_local["root"]),
        )
    test_csv_norm = normalize_csv_paths(
        csv_in=fold["test_csv"],
        csv_out=str(normalized_dir / "test.csv"),
        config_root=str(data_cfg_local["root"]),
    )

    if cv_protocol in {"4train_1eval", "esc50_4plus1", "pure_5fold"}:
        if val_csv_norm:
            train_csv_final = merge_csv_unique_by_path(
                train_csv_norm, val_csv_norm, str(normalized_dir / "train_4fold.csv")
            )
        else:
            train_csv_final = train_csv_norm
        val_csv_final = None
    elif cv_protocol in {"fold_val"}:
        # 5-fold CV with epoch-wise validation on the hold-out fold.
        # Train on 4 folds (train + optional val split), validate on fold test split.
        if val_csv_norm:
            train_csv_final = merge_csv_unique_by_path(
                train_csv_norm, val_csv_norm, str(normalized_dir / "train_4fold.csv")
            )
        else:
            train_csv_final = train_csv_norm
        val_csv_final = test_csv_norm
    else:
        train_csv_final = train_csv_norm
        val_csv_final = val_csv_norm

    # paths in normalized CSV are absolute; root value is ignored in Path(root) / abs_path.
    data_cfg_local["root"] = str(Path.cwd())
    train_set = build_dataset(data_cfg_local, model_cfg, train_csv_final, training=True)
    has_val = bool(val_csv_final)
    val_set = build_dataset(data_cfg_local, model_cfg, val_csv_final, training=False) if has_val else None
    test_set = build_dataset(data_cfg_local, model_cfg, test_csv_norm, training=False)

    train_loader = build_loader(train_set, batch_size=batch_size, num_workers=num_workers, training=True)
    val_loader = (
        build_loader(val_set, batch_size=batch_size, num_workers=num_workers, training=False)
        if has_val
        else None
    )
    test_loader = build_loader(test_set, batch_size=batch_size, num_workers=num_workers, training=False)

    num_classes = int(data_cfg_local.get("num_classes", infer_num_classes_from_fold(fold)))
    pooling_name = str(model_cfg.get("pooling", "ssrp_t")).lower()
    adaptive_return_mode = str(model_cfg.get("adaptive_return_mode", "z")).lower()
    use_alpha = pooling_name in {"adaptive_ssrp_t", "adaptive_ssrp"} and (
        log_alpha_stats or (entropy_lambda > 0.0) or adaptive_return_mode in {"alpha", "details"}
    )
    if use_alpha and adaptive_return_mode == "z":
        adaptive_return_mode = "alpha"

    model = PoolingAudioClassifier(    # model 설계
        backbone=model_cfg.get("backbone", "lightweight_cnn"),
        num_classes=num_classes,
        pooling=pooling_name,
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
        adaptive_use_branch_calibration=bool(model_cfg.get("adaptive_use_branch_calibration", True)),
        adaptive_return_mode=adaptive_return_mode,
        head_hidden=int(model_cfg.get("head_hidden", 128)),
        dropout=float(model_cfg.get("dropout", 0.5)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
        return_alpha=use_alpha,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=float(train_cfg.get("momentum", 0.9)),
            nesterov=bool(train_cfg.get("nesterov", False)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler = None
    if scheduler_name in {"reduce_on_plateau", "plateau", "reduce_lr_on_plateau"}:
        scheduler_mode = "min" if scheduler_monitor in {"val_loss", "train_loss"} else "max"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=scheduler_threshold,
            cooldown=scheduler_cooldown,
            min_lr=scheduler_min_lr,
        )
    elif scheduler_name in {"none", "off", "disabled"}:
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    artifact_tag = time.strftime("%Y%m%d_%H%M%S") if artifact_timestamp else None
    if checkpoint_timestamp:
        ckpt_name = f"best_{artifact_tag}.pt" if artifact_tag else f"best_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    else:
        ckpt_name = "best.pt"
    ckpt_path = fold_out / ckpt_name

    best_score = None
    best_epoch = 0
    best_val_metrics = None
    best_train_metrics = None
    history = []
    early_stop_bad_epochs = 0
    stopped_early = False
    stopped_epoch = None

    start = time.time()
    for epoch in range(1, epochs + 1):
        improved = False
        tr = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            num_classes=num_classes,
            mixup_alpha=mixup_alpha,
            entropy_lambda=entropy_lambda,
            show_pbar=use_tqdm,
            progress_desc=f"Train {fold['name']} E{epoch}/{epochs}",
        )
        va = None
        if has_val:
            va = evaluate(
                model,
                val_loader,
                device,
                criterion,
                num_classes=num_classes,
                collect_alpha_stats=False,
                show_pbar=use_tqdm,
                progress_desc=f"Val   {fold['name']} E{epoch}/{epochs}",
            )

        row = {"epoch": epoch, "train": tr}
        if va is not None:
            row["val"] = {k: v for k, v in va.items() if k != "alpha_stats"}
        history.append(row)

        if scheduler is not None:
            scheduler_score = _scheduler_score(scheduler_monitor, va, tr)
            if scheduler_score is not None:
                scheduler.step(scheduler_score)

        current_lr = float(optimizer.param_groups[0]["lr"])

        if has_val:
            current_score = _selection_score(best_metric, va)
            is_improved = False
            if best_score is None:
                is_improved = True
            elif best_metric == "val_loss":
                is_improved = current_score < (best_score - early_stopping_min_delta)
            else:
                is_improved = current_score > (best_score + early_stopping_min_delta)

            if is_improved:
                best_score = current_score
                best_epoch = epoch
                best_val_metrics = va
                improved = True
                early_stop_bad_epochs = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": cfg,
                        "fold": fold,
                        "num_classes": num_classes,
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                        "best_score": best_score,
                        "best_val_metrics": best_val_metrics,
                    },
                    ckpt_path,
                )
            else:
                early_stop_bad_epochs += 1
            if epoch_log:
                best_acc = float(best_val_metrics["accuracy"]) if best_val_metrics else float("nan")
                alpha_ent = tr.get("alpha_entropy")
                ent_msg = f" train_alpha_ent={float(alpha_ent):.4f}" if alpha_ent is not None else ""
                msg = (
                    f"[Epoch {epoch:03d}/{epochs}] fold={fold['name']} "
                    f"train_loss={float(tr['loss']):.4f} train_acc={float(tr['accuracy']):.4f} "
                    f"val_loss={float(va['loss']):.4f} val_acc={float(va['accuracy']):.4f} "
                    f"best_val_acc={best_acc:.4f}@{best_epoch} "
                    f"lr={current_lr:.6f}"
                )
                msg += ent_msg
                if improved:
                    msg += " *"
                if use_tqdm:
                    tqdm.write(msg)
                else:
                    print(msg)
            if early_stopping_patience > 0 and early_stop_bad_epochs >= early_stopping_patience:
                stopped_early = True
                stopped_epoch = epoch
                stop_msg = (
                    f"[EarlyStop] fold={fold['name']} epoch={epoch} "
                    f"no_improve={early_stop_bad_epochs} patience={early_stopping_patience}"
                )
                if use_tqdm:
                    tqdm.write(stop_msg)
                else:
                    print(stop_msg)
                break
        else:
            best_epoch = epoch
            best_metric = "final_epoch_no_val"
            best_score = float(tr["accuracy"])
            best_train_metrics = tr
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "fold": fold,
                    "num_classes": num_classes,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "best_score": best_score,
                    "best_train_metrics": best_train_metrics,
                },
                ckpt_path,
            )
            if epoch_log:
                alpha_ent = tr.get("alpha_entropy")
                ent_msg = f" train_alpha_ent={float(alpha_ent):.4f}" if alpha_ent is not None else ""
                msg = (
                    f"[Epoch {epoch:03d}/{epochs}] fold={fold['name']} "
                    f"train_loss={float(tr['loss']):.4f} train_acc={float(tr['accuracy']):.4f} "
                    f"(no val; final-epoch selection) "
                    f"lr={current_lr:.6f}"
                )
                msg += ent_msg
                if use_tqdm:
                    tqdm.write(msg)
                else:
                    print(msg)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"], strict=True)

    te = evaluate(
        model,
        test_loader,
        device,
        criterion,
        num_classes=num_classes,
        collect_alpha_stats=use_alpha,
        show_pbar=use_tqdm,
        progress_desc=f"Test  {fold['name']}",
    )
    input_shape = infer_input_shape(test_set)

    total_params, trainable_params = count_parameters(model)
    bb_total, bb_trainable = count_parameters(model.backbone)
    pool_total, pool_trainable = count_parameters(model.pool)
    latency = measure_latency_ms(
        model=model,
        input_shape=input_shape,
        device=device,
        warmup=int(cfg["train"].get("latency_warmup", 10)),
        runs=int(cfg["train"].get("latency_runs", 50)),
    )
    flops = try_compute_flops(model=model, input_shape=input_shape, device=device)

    result = {
        "fold": fold["name"],
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_score": best_score,
        "checkpoint_path": str(ckpt_path),
        "best_val_accuracy": float(best_val_metrics["accuracy"]) if best_val_metrics else None,
        "best_val_macro_f1": float(best_val_metrics["macro_f1"]) if best_val_metrics else None,
        "best_val_loss": float(best_val_metrics["loss"]) if best_val_metrics else None,
        "selection_mode": "val_based" if has_val else "final_epoch_no_val",
        "train_protocol": cv_protocol,
        "final_train_accuracy": float(best_train_metrics["accuracy"]) if best_train_metrics else None,
        "final_train_macro_f1": float(best_train_metrics["macro_f1"]) if best_train_metrics else None,
        "test": {
            "accuracy": float(te["accuracy"]),
            "macro_precision": float(te["macro_precision"]),
            "macro_recall": float(te["macro_recall"]),
            "macro_f1": float(te["macro_f1"]),
            "loss": float(te["loss"]),
        },
        "pooling_feature_time_lengths": te.get("feature_time_lengths", []),
        "adaptive_alpha_stats": te.get("alpha_stats", None),
        "stopped_early": bool(stopped_early),
        "stopped_epoch": int(stopped_epoch) if stopped_epoch is not None else None,
        "efficiency": {
            "params_total": total_params,
            "params_trainable": trainable_params,
            "params_backbone_total": bb_total,
            "params_backbone_trainable": bb_trainable,
            "params_pool_total": pool_total,
            "params_pool_trainable": pool_trainable,
            "input_shape": list(input_shape),
            **latency,
            **flops,
        },
        "elapsed_sec": float(time.time() - start),
    }

    history_json_name = f"history_{artifact_tag}.json" if artifact_tag else "history.json"
    history_csv_name = f"history_{artifact_tag}.csv" if artifact_tag else "history.csv"
    result_json_name = f"result_{artifact_tag}.json" if artifact_tag else "result.json"

    history_json_path = fold_out / history_json_name
    history_csv_path = fold_out / history_csv_name
    result_json_path = fold_out / result_json_name

    result["artifact_tag"] = artifact_tag
    result["history_json_path"] = str(history_json_path)
    result["history_csv_path"] = str(history_csv_path)
    result["result_json_path"] = str(result_json_path)

    history_json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    history_csv_path.write_text(_history_to_csv(history), encoding="utf-8")
    result_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Keep stable filenames for compatibility with existing tooling.
    if history_json_name != "history.json":
        (fold_out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if history_csv_name != "history.csv":
        (fold_out / "history.csv").write_text(_history_to_csv(history), encoding="utf-8")
    if result_json_name != "result.json":
        (fold_out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _history_to_csv(history: List[Dict]) -> str:
    lines = [
        "epoch,train_loss,train_acc,train_macro_f1,train_alpha_entropy,val_loss,val_acc,val_macro_f1,train_T,val_T"
    ]
    for row in history:
        tr = row["train"]
        va = row.get("val")
        va_loss = float(va["loss"]) if va else float("nan")
        va_acc = float(va["accuracy"]) if va else float("nan")
        va_f1 = float(va["macro_f1"]) if va else float("nan")
        tr_ent = float(tr["alpha_entropy"]) if tr.get("alpha_entropy") is not None else float("nan")
        va_t = ";".join(str(v) for v in (va.get("feature_time_lengths", []) if va else []))
        lines.append(
            "{epoch},{tr_loss:.6f},{tr_acc:.6f},{tr_f1:.6f},{tr_ent:.6f},{va_loss:.6f},{va_acc:.6f},{va_f1:.6f},\"{tr_t}\",\"{va_t}\"".format(
                epoch=int(row["epoch"]),
                tr_loss=float(tr["loss"]),
                tr_acc=float(tr["accuracy"]),
                tr_f1=float(tr["macro_f1"]),
                tr_ent=tr_ent,
                va_loss=va_loss,
                va_acc=va_acc,
                va_f1=va_f1,
                tr_t=";".join(str(v) for v in tr.get("feature_time_lengths", [])),
                va_t=va_t,
            )
        )
    return "\n".join(lines) + "\n"


def summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    test_acc = np.array([float(r["test"]["accuracy"]) for r in results], dtype=np.float64)
    test_f1 = np.array([float(r["test"]["macro_f1"]) for r in results], dtype=np.float64)
    val_acc_vals = [r["best_val_accuracy"] for r in results if r.get("best_val_accuracy") is not None]
    val_acc = np.array([float(v) for v in val_acc_vals], dtype=np.float64) if val_acc_vals else None
    latency = np.array([float(r["efficiency"]["latency_ms_mean"]) for r in results], dtype=np.float64)

    out = {
        "folds": len(results),
        "test_accuracy_mean": float(test_acc.mean()),
        "test_accuracy_std": float(test_acc.std()),
        "test_macro_f1_mean": float(test_f1.mean()),
        "test_macro_f1_std": float(test_f1.std()),
        "latency_ms_mean": float(latency.mean()),
        "latency_ms_std": float(latency.std()),
    }
    if val_acc is not None and val_acc.size > 0:
        out["best_val_accuracy_mean"] = float(val_acc.mean())
        out["best_val_accuracy_std"] = float(val_acc.std())
    else:
        out["best_val_accuracy_mean"] = None
        out["best_val_accuracy_std"] = None
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ESC-50 pooling tradeoff CV runner (fixed backbone + pooling variants)."
    )
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--root", default=None, help="Override data.root")
    p.add_argument("--folds-dir", default=None, help="Override data.folds_dir")
    p.add_argument("--out-dir", default=None, help="Override output.out_dir")
    p.add_argument("--run-name", default=None, help="Output run name")
    p.add_argument("--backbone", default=None, help="Override model.backbone")
    p.add_argument("--pooling", default=None, help="Override model.pooling")
    p.add_argument("--ssrp-w", type=int, default=None)
    p.add_argument("--ssrp-k", type=int, default=None)
    p.add_argument("--ssrp-b-k", type=int, default=None)
    p.add_argument("--adaptive-ks", default=None, help="Comma-separated ints, e.g. 4,8,12")
    p.add_argument("--adaptive-gate-hidden", type=int, default=None)
    p.add_argument(
        "--best-metric",
        default=None,
        choices=["val_accuracy", "val_macro_f1", "val_loss"],
        help="Model selection metric",
    )
    p.add_argument(
        "--cv-protocol",
        default=None,
        choices=["with_val", "4train_1eval", "esc50_4plus1", "pure_5fold", "fold_val"],
        help="CV protocol: with_val, pure_5fold, or fold_val (epoch-wise val on hold-out fold).",
    )
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--early-stopping-min-delta", type=float, default=None)
    p.add_argument(
        "--input-representation",
        default=None,
        choices=["mel", "waveform"],
        help="Override model.input_representation",
    )
    p.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone weights")
    p.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-folds", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = merge_config(cfg, args)
    run_tag = time.strftime("%Y%m%d_%H%M%S")

    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    device = resolve_device(args.device)
    folds = discover_folds(cfg["data"]["folds_dir"])
    if args.max_folds:
        folds = folds[: int(args.max_folds)]

    out_dir = Path(cfg["output"].get("out_dir", "outputs/pooling_tradeoff"))
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = cfg["model"].get("backbone", "lightweight_cnn")
    pooling = cfg["model"].get("pooling", "ssrp_t")
    run_name = cfg["output"].get("run_name") or f"{backbone}__{pooling}__seed{seed}"
    run_out_dir = out_dir / run_name
    run_out_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    def _fmt(v):
        return "NA" if v is None else f"{float(v):.4f}"
    for fold in folds:
        print(f"\n[Fold] {fold['name']} | run={run_name} | pooling={pooling} | seed={seed}")
        fold_result = train_one_fold(fold=fold, cfg=cfg, device=device, run_out_dir=run_out_dir)
        fold_results.append(fold_result)
        print(
            f"  best_val_acc={_fmt(fold_result['best_val_accuracy'])} "
            f"test_acc={_fmt(fold_result['test']['accuracy'])} "
            f"test_f1={_fmt(fold_result['test']['macro_f1'])} "
            f"T={fold_result['pooling_feature_time_lengths']}"
        )
        alpha_stats = fold_result.get("adaptive_alpha_stats")
        if alpha_stats:
            ent_n = alpha_stats.get("alpha_entropy_norm_mean")
            dom_r = alpha_stats.get("alpha_argmax_dominant_ratio")
            cond = alpha_stats.get("temporal_conditioning") or {}
            short_a = ((cond.get("short_term") or {}).get("alpha_mean"))
            long_a = ((cond.get("long_term") or {}).get("alpha_mean"))
            short_long_gap = None
            if short_a is not None and long_a is not None:
                short_long_gap = float(np.mean(np.abs(np.asarray(short_a) - np.asarray(long_a))))
            print(
                f"  alpha_entropy_norm={_fmt(ent_n)} "
                f"alpha_dominant_ratio={_fmt(dom_r)} "
                f"alpha_short_long_gap={_fmt(short_long_gap)}"
            )

    summary = summarize(fold_results)
    summary_payload = {
        "config": cfg,
        "run_tag": run_tag,
        "run_name": run_name,
        "backbone": backbone,
        "pooling": pooling,
        "seed": seed,
        "device": str(device),
        "fold_results": fold_results,
        "summary": summary,
    }

    summary_json = run_out_dir / "cv_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    summary_csv = run_out_dir / "cv_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "best_val_accuracy",
                "test_accuracy",
                "test_macro_f1",
                "latency_ms_mean",
                "pooling_feature_time_lengths",
                "params_total",
                "params_backbone_total",
                "params_pool_total",
                "flops",
                "macs",
            ],
        )
        writer.writeheader()
        for r in fold_results:
            writer.writerow(
                {
                    "fold": r["fold"],
                    "best_val_accuracy": r["best_val_accuracy"],
                    "test_accuracy": r["test"]["accuracy"],
                    "test_macro_f1": r["test"]["macro_f1"],
                    "latency_ms_mean": r["efficiency"]["latency_ms_mean"],
                    "pooling_feature_time_lengths": ";".join(str(v) for v in r["pooling_feature_time_lengths"]),
                    "params_total": r["efficiency"]["params_total"],
                    "params_backbone_total": r["efficiency"]["params_backbone_total"],
                    "params_pool_total": r["efficiency"]["params_pool_total"],
                    "flops": r["efficiency"]["flops"],
                    "macs": r["efficiency"]["macs"],
                }
            )

    # Save run parameter snapshots.
    save_params_timestamp = bool(cfg.get("train", {}).get("save_params_timestamp", True))
    resolved_config_stable = run_out_dir / "resolved_config.yaml"
    resolved_config_stable.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    resolved_config_path = resolved_config_stable
    if save_params_timestamp:
        resolved_config_path = run_out_dir / f"resolved_config_{run_tag}.yaml"
        resolved_config_path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )

    run_params_payload = {
        "run_tag": run_tag,
        "run_name": run_name,
        "seed": seed,
        "device": str(device),
        "config_path_arg": str(args.config),
        "resolved_config_path": str(resolved_config_path),
        "cv_protocol": str(cfg.get("train", {}).get("cv_protocol", "with_val")),
        "best_metric": str(cfg.get("train", {}).get("best_metric", "val_accuracy")),
        "config": cfg,
    }
    run_params_path = run_out_dir / f"run_params_{run_tag}.json"
    run_params_path.write_text(json.dumps(run_params_payload, indent=2), encoding="utf-8")

    summary_payload["resolved_config_path"] = str(resolved_config_path)
    summary_payload["run_params_path"] = str(run_params_path)
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n[Done]")
    print(f"summary_json: {summary_json}")
    print(f"summary_csv : {summary_csv}")
    print(f"params_yaml : {resolved_config_path}")
    print(f"params_json : {run_params_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
