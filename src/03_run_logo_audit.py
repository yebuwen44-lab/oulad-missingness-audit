#!/usr/bin/env python3
"""Leave-one-presentation-out audit with native-NaN and top-k diagnostics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis_core import make_pipeline, representation_columns


def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=root / "data/processed/oulad_week4_analysis_table.csv")
    p.add_argument("--config", type=Path, default=root / "config/feature_sets_week4.json")
    p.add_argument("--output-dir", type=Path, default=root / "results/rerun/logo")
    p.add_argument("--models", nargs="+", default=["lr", "lightgbm", "xgboost"], choices=["lr", "lightgbm", "xgboost"])
    p.add_argument("--representations", nargs="+", default=["F1_median", "F1_native", "F2", "F3"])
    p.add_argument("--bootstrap-iterations", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=20260618)
    return p.parse_args()


def topk(y: np.ndarray, p: np.ndarray, fraction: float):
    k = max(1, int(math.ceil(len(y) * fraction)))
    order = np.argsort(-p)[:k]
    tp = int(y[order].sum())
    precision = tp / k
    recall = tp / int(y.sum()) if int(y.sum()) else float("nan")
    prevalence = float(y.mean())
    lift = precision / prevalence if prevalence else float("nan")
    return precision, recall, lift, k


def paired_bootstrap(values: np.ndarray, iterations: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(iterations, dtype=float)
    for i in range(iterations):
        boot[i] = values[rng.integers(0, n, n)].mean()
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))["w4"]
    df = df[df[cfg["active_filter_col"]] == 1].copy()
    df["held_out_presentation"] = df["code_module"].astype(str) + "__" + df["code_presentation"].astype(str)
    group_names = sorted(df["held_out_presentation"].unique())
    representation_map = representation_columns(cfg)
    detail_rows, top_rows = [], []

    for held_out in group_names:
        test_mask = df["held_out_presentation"] == held_out
        train_mask = ~test_mask
        y_train = df.loc[train_mask, "y_dropout"].astype(int)
        y_test = df.loc[test_mask, "y_dropout"].astype(int)

        for representation in args.representations:
            columns = representation_map[representation]
            for model_name in args.models:
                if representation == "F1_native" and model_name == "lr":
                    continue
                pipeline = make_pipeline(df, columns, model_name, representation, y_train)
                pipeline.fit(df.loc[train_mask, columns], y_train)
                probability = pipeline.predict_proba(df.loc[test_mask, columns])[:, 1]
                detail_rows.append(
                    {
                        "held_out_presentation": held_out,
                        "n": int(test_mask.sum()),
                        "dropout_rate": float(y_test.mean()),
                        "model": model_name,
                        "representation": representation,
                        "auprc": average_precision_score(y_test, probability),
                        "auroc": roc_auc_score(y_test, probability),
                    }
                )
                for fraction in (0.10, 0.20):
                    precision, recall, lift, k = topk(y_test.to_numpy(), probability, fraction)
                    top_rows.append(
                        {
                            "held_out_presentation": held_out,
                            "model": model_name,
                            "representation": representation,
                            "top_fraction": fraction,
                            "k": k,
                            "precision": precision,
                            "recall": recall,
                            "lift": lift,
                        }
                    )
                print(f"completed {held_out}/{representation}/{model_name}", flush=True)

    detail = pd.DataFrame(detail_rows)
    top = pd.DataFrame(top_rows)
    macro = (
        detail.groupby(["model", "representation"], as_index=False)
        .agg(macro_auprc=("auprc", "mean"), sd_auprc=("auprc", "std"), macro_auroc=("auroc", "mean"), n_groups=("held_out_presentation", "nunique"))
    )
    top_macro = (
        top.groupby(["model", "representation", "top_fraction"], as_index=False)
        .agg(precision=("precision", "mean"), recall=("recall", "mean"), lift=("lift", "mean"), n_groups=("held_out_presentation", "nunique"))
    )

    contrast_rows = []
    for model_name in detail["model"].unique():
        pivot = detail[detail["model"] == model_name].pivot(index="held_out_presentation", columns="representation", values="auprc")
        if "F3" not in pivot:
            continue
        for comparator in [c for c in ["F1_median", "F1_native", "F2"] if c in pivot]:
            delta = (pivot["F3"] - pivot[comparator]).dropna().to_numpy()
            low, high = paired_bootstrap(delta, args.bootstrap_iterations, args.bootstrap_seed)
            try:
                w = float(wilcoxon(delta, alternative="greater").pvalue)
            except ValueError:
                w = float("nan")
            positive = int((delta > 0).sum())
            contrast_rows.append(
                {
                    "model": model_name,
                    "contrast": f"F3-{comparator}",
                    "mean_delta_auprc": float(delta.mean()),
                    "positive_groups": positive,
                    "n_groups": len(delta),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "wilcoxon_one_sided_p": w,
                    "exact_sign_test_one_sided_p": float(binomtest(positive, len(delta), 0.5, alternative="greater").pvalue),
                }
            )

    detail.to_csv(args.output_dir / "logo_presentation_results.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(args.output_dir / "logo_macro_summary.csv", index=False, encoding="utf-8-sig")
    top.to_csv(args.output_dir / "logo_topk_presentation_results.csv", index=False, encoding="utf-8-sig")
    top_macro.to_csv(args.output_dir / "logo_topk_macro_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(contrast_rows).to_csv(args.output_dir / "logo_paired_contrasts.csv", index=False, encoding="utf-8-sig")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
