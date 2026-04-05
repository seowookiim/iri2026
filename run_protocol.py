from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def make_variants(include_ssrp_b: bool, minimal_set: bool) -> List[Dict]:
    p0 = {"id": "P0", "label": "GAP", "pooling": "gap"}
    p1 = {"id": "P1", "label": "GMP", "pooling": "gmp"}
    p2 = {"id": "P2", "label": "ASP", "pooling": "asp"}
    p3 = {"id": "P3", "label": "SSRP-T(W4,K12)", "pooling": "ssrp_t", "ssrp_w": 4, "ssrp_k": 12}
    p4 = {"id": "P4", "label": "SSRP-T(W2,K4)", "pooling": "ssrp_t", "ssrp_w": 2, "ssrp_k": 4}
    p5 = {"id": "P5", "label": "SSRP-B(W4)", "pooling": "ssrp_b", "ssrp_w": 4, "ssrp_b_k": 12}
    p6 = {
        "id": "P6",
        "label": "AdaptiveSSRP-T(W4,Ks=4/8/12,h128)",
        "pooling": "adaptive_ssrp_t",
        "ssrp_w": 4,
        "adaptive_ks": [4, 8, 12],
        "adaptive_gate_hidden": 128,
    }
    p7 = {
        "id": "P7",
        "label": "AdaptiveSSRP-T(W2,Ks=2/4/6,h64)",
        "pooling": "adaptive_ssrp_t",
        "ssrp_w": 2,
        "adaptive_ks": [2, 4, 6],
        "adaptive_gate_hidden": 64,
    }

    if minimal_set:
        return [p0, p1, p2, p3, p6]

    variants = [p0, p1, p2, p3, p4, p6, p7]
    if include_ssrp_b:
        variants.insert(5, p5)
    return variants


def make_ssrp_adaptive_compare_variants(
    baseline_pooling: str,
    model_cfg: Dict,
) -> List[Dict]:
    baseline_pooling = (baseline_pooling or "gap").lower()
    if baseline_pooling not in {"gap", "gmp", "current"}:
        raise ValueError(f"Unsupported baseline_pooling: {baseline_pooling}")

    variants: List[Dict] = []
    if baseline_pooling == "current":
        p = (model_cfg.get("pooling", "gap") or "gap").lower()
        base = {"id": "P0", "label": f"Baseline({p.upper()})", "pooling": p}
        if p == "ssrp_t":
            base["ssrp_w"] = int(model_cfg.get("ssrp_w", 4))
            base["ssrp_k"] = int(model_cfg.get("ssrp_k", 12))
        elif p == "ssrp_b":
            base["ssrp_w"] = int(model_cfg.get("ssrp_w", 4))
            base["ssrp_b_k"] = int(model_cfg.get("ssrp_b_k", 12))
        elif p == "adaptive_ssrp_t":
            base["ssrp_w"] = int(model_cfg.get("ssrp_w", 4))
            base["adaptive_ks"] = list(model_cfg.get("adaptive_ks", [4, 8, 12]))
            base["adaptive_gate_hidden"] = int(model_cfg.get("adaptive_gate_hidden", 128))
        variants.append(base)
    elif baseline_pooling == "gmp":
        variants.append({"id": "P0", "label": "Baseline(GMP)", "pooling": "gmp"})
    else:
        variants.append({"id": "P0", "label": "Baseline(GAP)", "pooling": "gap"})

    w = int(model_cfg.get("ssrp_w", 4))
    variants.extend(
        [
            {"id": "P1", "label": "ASP", "pooling": "asp"},
            {"id": "P2", "label": f"SSRP-T(W{w},K4)", "pooling": "ssrp_t", "ssrp_w": w, "ssrp_k": 4},
            {"id": "P3", "label": f"SSRP-T(W{w},K8)", "pooling": "ssrp_t", "ssrp_w": w, "ssrp_k": 8},
            {"id": "P4", "label": f"SSRP-T(W{w},K12)", "pooling": "ssrp_t", "ssrp_w": w, "ssrp_k": 12},
            {
                "id": "P5",
                "label": f"AdaptiveSSRP-T(W{w},Ks=4/8/12,h{int(model_cfg.get('adaptive_gate_hidden', 128))})",
                "pooling": "adaptive_ssrp_t",
                "ssrp_w": w,
                "adaptive_ks": [4, 8, 12],
                "adaptive_gate_hidden": int(model_cfg.get("adaptive_gate_hidden", 128)),
            },
        ]
    )
    return variants


def make_adaptive_only_variants(model_cfg: Dict) -> List[Dict]:
    w = int(model_cfg.get("ssrp_w", 4))
    ks = [int(k) for k in model_cfg.get("adaptive_ks", [4, 8, 12])]
    h = int(model_cfg.get("adaptive_gate_hidden", 128))
    return [
        {
            "id": "P4",
            "label": f"AdaptiveSSRP-T(W{w},Ks={'/'.join(str(k) for k in ks)},h{h})",
            "pooling": "adaptive_ssrp_t",
            "ssrp_w": w,
            "adaptive_ks": ks,
            "adaptive_gate_hidden": h,
        }
    ]


def make_run_name(variant: Dict, seed: int, model_cfg: Dict) -> str:
    if variant["pooling"] == "gap":
        return f"esc50_lwcnn_pool_GAP_seed{seed}"
    if variant["pooling"] == "gmp":
        return f"esc50_lwcnn_pool_GMP_seed{seed}"
    if variant["pooling"] == "asp":
        return f"esc50_lwcnn_pool_ASP_seed{seed}"
    if variant["pooling"] == "ssrp_t":
        return f"esc50_lwcnn_pool_SSRPT_W{variant['ssrp_w']}K{variant['ssrp_k']}_seed{seed}"
    if variant["pooling"] == "ssrp_b":
        return f"esc50_lwcnn_pool_SSRPB_W{variant['ssrp_w']}_seed{seed}"
    if variant["pooling"] == "adaptive_ssrp_t":
        ks = "".join(str(k) for k in variant["adaptive_ks"])
        return (
            f"esc50_lwcnn_pool_AdaptSSRP_W{variant['ssrp_w']}_"
            f"Ks{ks}_h{variant['adaptive_gate_hidden']}_seed{seed}"
        )
    raise ValueError(f"Unknown pooling variant: {variant}")


def run_one(
    config: Path,
    out_dir: Path,
    device: str,
    seed: int,
    variant: Dict,
    model_cfg: Dict,
    cv_protocol: str = "fold_val",
    root: str | None = None,
    folds_dir: str | None = None,
    max_folds: int | None = None,
) -> Path:
    run_name = make_run_name(variant, seed, model_cfg)
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "experiments.esc50_pooling_tradeoff.train_eval_cv",
        "--config",
        str(config),
        "--out-dir",
        str(out_dir),
        "--run-name",
        run_name,
        "--device",
        device,
        "--seed",
        str(seed),
        "--backbone",
        "lightweight_cnn",
        "--cv-protocol",
        cv_protocol,
        "--pooling",
        variant["pooling"],
    ]
    if root:
        cmd += ["--root", root]
    if folds_dir:
        cmd += ["--folds-dir", folds_dir]
    if max_folds is not None:
        cmd += ["--max-folds", str(max_folds)]
    if "ssrp_w" in variant:
        cmd += ["--ssrp-w", str(variant["ssrp_w"])]
    if "ssrp_k" in variant:
        cmd += ["--ssrp-k", str(variant["ssrp_k"])]
    if "ssrp_b_k" in variant:
        cmd += ["--ssrp-b-k", str(variant["ssrp_b_k"])]
    if "adaptive_ks" in variant:
        cmd += ["--adaptive-ks", ",".join(str(x) for x in variant["adaptive_ks"])]
    if "adaptive_gate_hidden" in variant:
        cmd += ["--adaptive-gate-hidden", str(variant["adaptive_gate_hidden"])]

    print("\n" + "=" * 100)
    print(f"[RUN] {run_name}")
    print(" ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)
    return out_dir / run_name / "cv_summary.json"


def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean * 100.0:.2f}+-{std * 100.0:.2f}"


def aggregate(results: List[Dict]) -> Dict:
    seed_fold_means_acc = np.array([r["summary"]["test_accuracy_mean"] for r in results], dtype=np.float64)
    seed_fold_means_f1 = np.array([r["summary"]["test_macro_f1_mean"] for r in results], dtype=np.float64)

    all_fold_acc = []
    all_fold_f1 = []
    all_best_val = []
    params_total = []
    pool_params = []
    flops = []
    for r in results:
        for fr in r["fold_results"]:
            all_fold_acc.append(float(fr["test"]["accuracy"]))
            all_fold_f1.append(float(fr["test"]["macro_f1"]))
            if fr.get("best_val_accuracy") is not None:
                all_best_val.append(float(fr["best_val_accuracy"]))
            params_total.append(float(fr["efficiency"]["params_total"]))
            pool_params.append(float(fr["efficiency"]["params_pool_total"]))
            if fr["efficiency"]["flops"] is not None:
                flops.append(float(fr["efficiency"]["flops"]))

    payload = {
        "seed_count": len(results),
        "seed_level": {
            "acc_mean_of_fold_mean": float(seed_fold_means_acc.mean()),
            "acc_std_of_fold_mean": float(seed_fold_means_acc.std()),
            "f1_mean_of_fold_mean": float(seed_fold_means_f1.mean()),
            "f1_std_of_fold_mean": float(seed_fold_means_f1.std()),
        },
        "overall_fold_seed": {
            "best_val_acc_mean": float(np.mean(all_best_val)) if all_best_val else None,
            "best_val_acc_std": float(np.std(all_best_val)) if all_best_val else None,
            "acc_mean": float(np.mean(all_fold_acc)),
            "acc_std": float(np.std(all_fold_acc)),
            "f1_mean": float(np.mean(all_fold_f1)),
            "f1_std": float(np.std(all_fold_f1)),
        },
        "efficiency": {
            "params_total_mean": float(np.mean(params_total)),
            "pool_params_mean": float(np.mean(pool_params)),
            "flops_mean": float(np.mean(flops)) if flops else None,
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Protocol runner: fixed lightweight CNN + pooling-only comparison on ESC-50."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--variant-set",
        default="full",
        choices=["full", "minimal", "ssrp_adaptive_compare", "adaptive_only"],
        help="Experiment variant set.",
    )
    parser.add_argument(
        "--baseline-pooling",
        default="gap",
        choices=["gap", "gmp", "current"],
        help="Baseline pooling for ssrp_adaptive_compare.",
    )
    parser.add_argument(
        "--cv-protocol",
        default="fold_val",
        choices=["fold_val", "pure_5fold", "with_val"],
        help="fold_val: epoch-wise validation on hold-out fold.",
    )
    parser.add_argument("--root", default=None)
    parser.add_argument("--folds-dir", default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--ssrp-w", type=int, default=None, help="Override model.ssrp_w for variants.")
    parser.add_argument(
        "--adaptive-ks",
        default=None,
        help="Override model.adaptive_ks as comma-separated ints, e.g. 2,4,6.",
    )
    parser.add_argument(
        "--adaptive-gate-hidden",
        type=int,
        default=None,
        help="Override model.adaptive_gate_hidden for adaptive variants.",
    )
    parser.add_argument(
        "--result-tag",
        default=None,
        help="Tag suffix for protocol output filenames. Default: current timestamp (YYYYMMDD_HHMMSS).",
    )
    parser.add_argument("--include-ssrp-b", action="store_true")
    parser.add_argument("--minimal-set", action="store_true")
    args = parser.parse_args()

    config = Path(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_int_list(args.seeds)
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    model_cfg = (cfg.get("model", {}) or {})

    # Optional model overrides from CLI for reproducible run-name / variant generation.
    if args.ssrp_w is not None:
        model_cfg["ssrp_w"] = int(args.ssrp_w)
    if args.adaptive_ks is not None:
        model_cfg["adaptive_ks"] = parse_int_list(args.adaptive_ks)
    if args.adaptive_gate_hidden is not None:
        model_cfg["adaptive_gate_hidden"] = int(args.adaptive_gate_hidden)

    if args.variant_set == "adaptive_only":
        variants = make_adaptive_only_variants(model_cfg=model_cfg)
    elif args.variant_set == "ssrp_adaptive_compare":
        variants = make_ssrp_adaptive_compare_variants(
            baseline_pooling=args.baseline_pooling,
            model_cfg=model_cfg,
        )
    elif args.variant_set == "minimal" or args.minimal_set:
        variants = make_variants(include_ssrp_b=False, minimal_set=True)
    else:
        variants = make_variants(include_ssrp_b=args.include_ssrp_b, minimal_set=False)

    result_tag = str(args.result_tag).strip() if args.result_tag else time.strftime("%Y%m%d_%H%M%S")

    protocol = {
        "config": str(config),
        "out_dir": str(out_dir),
        "device": args.device,
        "seeds": seeds,
        "cv_protocol": args.cv_protocol,
        "variant_set": args.variant_set,
        "baseline_pooling": args.baseline_pooling,
        "model_overrides": {
            "ssrp_w": args.ssrp_w,
            "adaptive_ks": args.adaptive_ks,
            "adaptive_gate_hidden": args.adaptive_gate_hidden,
        },
        "result_tag": result_tag,
        "variants": variants,
    }
    protocol_path = out_dir / f"protocol_{result_tag}.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    variant_runs = []
    for variant in variants:
        seed_runs = []
        for seed in seeds:
            summary_path = run_one(
                config=config,
                out_dir=out_dir,
                device=args.device,
                seed=seed,
                variant=variant,
                model_cfg=model_cfg,
                cv_protocol=args.cv_protocol,
                root=args.root,
                folds_dir=args.folds_dir,
                max_folds=args.max_folds,
            )
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            seed_runs.append(payload)
        variant_runs.append({"variant": variant, "seed_runs": seed_runs, "aggregate": aggregate(seed_runs)})

    baseline = None
    for item in variant_runs:
        if item["variant"]["pooling"] == "gap":
            baseline = item["aggregate"]["efficiency"]
            break

    table_rows = []
    for item in variant_runs:
        var = item["variant"]
        agg = item["aggregate"]
        eff = agg["efficiency"]
        delta_params = None
        delta_flops = None
        if baseline is not None:
            delta_params = eff["params_total_mean"] - baseline["params_total_mean"]
            if eff["flops_mean"] is not None and baseline["flops_mean"] is not None:
                delta_flops = eff["flops_mean"] - baseline["flops_mean"]

        best_val_mean = agg["overall_fold_seed"]["best_val_acc_mean"]
        best_val_std = agg["overall_fold_seed"]["best_val_acc_std"]
        table_rows.append(
            {
                "pooling_id": var["id"],
                "pooling": var["label"],
                "best_val_acc_mean_std_pct": (
                    _fmt_mean_std(best_val_mean, best_val_std)
                    if best_val_mean is not None and best_val_std is not None
                    else "NA"
                ),
                "acc_mean_std_pct": _fmt_mean_std(
                    agg["overall_fold_seed"]["acc_mean"], agg["overall_fold_seed"]["acc_std"]
                ),
                "macro_f1_mean_std_pct": _fmt_mean_std(
                    agg["overall_fold_seed"]["f1_mean"], agg["overall_fold_seed"]["f1_std"]
                ),
                "params_delta": delta_params,
                "extra_ops_flops_delta": delta_flops,
            }
        )

    final_payload = {"result_tag": result_tag, "variants": variant_runs, "table1": table_rows}
    protocol_results_path = out_dir / f"protocol_results_{result_tag}.json"
    protocol_results_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")

    csv_path = out_dir / f"table1_pooling_comparison_{result_tag}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(
            "PoolingID,Pooling,BestValAcc(mean+-std)%,Acc(mean+-std)%,MacroF1(mean+-std)%,ParamsDelta,ExtraOpsFLOPsDelta\n"
        )
        for row in table_rows:
            f.write(
                "{},{},{},{},{},{},{}\n".format(
                    row["pooling_id"],
                    row["pooling"],
                    row["best_val_acc_mean_std_pct"],
                    row["acc_mean_std_pct"],
                    row["macro_f1_mean_std_pct"],
                    row["params_delta"],
                    row["extra_ops_flops_delta"],
                )
            )

    md_path = out_dir / f"table1_pooling_comparison_{result_tag}.md"
    md_lines = [
        "Table 1) Pooling comparison on ESC-50 (fixed lightweight CNN)",
        "",
        "| Pooling | Best Val Acc(mean+-std) | Acc(mean+-std) | Macro-F1(mean+-std) | Params delta | Extra ops (FLOPs delta) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        md_lines.append(
            f"| {row['pooling']} | {row['best_val_acc_mean_std_pct']} | {row['acc_mean_std_pct']} | {row['macro_f1_mean_std_pct']} | "
            f"{row['params_delta']} | {row['extra_ops_flops_delta']} |"
        )
    md_lines.append("")
    md_lines.append("Aggregation: fold mean -> seed mean -> overall (fold x seed) mean/std.")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n[Done]")
    print(f"protocol        : {protocol_path}")
    print(f"protocol_results: {protocol_results_path}")
    print(f"table_csv       : {csv_path}")
    print(f"table_md        : {md_path}")


if __name__ == "__main__":
    main()
