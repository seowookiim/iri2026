from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch


def count_parameters(module: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def classification_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    y_true = y_true.to(torch.long).view(-1)
    y_pred = y_pred.to(torch.long).view(-1)
    if y_true.numel() == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    conf = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1.0

    tp = torch.diag(conf)
    pred_pos = conf.sum(dim=0)
    true_pos = conf.sum(dim=1)

    eps = 1e-12
    precision = tp / (pred_pos + eps)
    recall = tp / (true_pos + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    acc = float((tp.sum() / conf.sum().clamp_min(1.0)).item())
    return {
        "accuracy": acc,
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
    }


@torch.inference_mode()
def measure_latency_ms(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    warmup: int = 10,
    runs: int = 50,
) -> Dict[str, float]:
    model = model.to(device)
    model.eval()
    x = torch.randn(*input_shape, device=device)

    for _ in range(max(1, int(warmup))):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    samples = []
    for _ in range(max(1, int(runs))):
        start = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)

    t = torch.tensor(samples, dtype=torch.float64)
    return {
        "latency_ms_mean": float(t.mean().item()),
        "latency_ms_std": float(t.std(unbiased=False).item()),
        "latency_ms_p50": float(t.quantile(0.50).item()),
        "latency_ms_p95": float(t.quantile(0.95).item()),
    }


def try_compute_flops(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> Dict[str, Optional[float]]:
    try:
        from thop import profile  # type: ignore
    except Exception:
        return {"flops": None, "macs": None}

    x = torch.randn(*input_shape, device=device)
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        macs, _ = profile(model, inputs=(x,), verbose=False)
    return {"flops": float(macs) * 2.0, "macs": float(macs)}

