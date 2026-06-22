#!/usr/bin/env python3
"""Portable five-fold GroupKFold reproduction for the Week-4 active-eligible cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from analysis_core import make_pipeline, representation_columns


def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=root / "data/processed/oulad_week4_analysis_table.csv")
    p.add_argument("--config", type=Path, default=root / "config/feature_sets_week4.json")
    p.add_argument("--output-dir", type=Path, default=root / "results/rerun/groupkfold")
    p.add_argument("--models", nargs="+", default=["lr", "lightgbm", "xgboost"], choices=["lr", "lightgbm", "xgboost"])
    p.add_argument(
        "--representations",
        nargs="+",
        default=["F1", "F1_median", "F2", "F3", "F4_all", "F4_structural_only", "F4_behavioral_only"],
    )
    return p.parse_args()


def metrics(y_true, probability):
    prediction = (probability >= 0.5).astype(int)
    return {
        "auprc": average_precision_score(y_true, probability),
        "auroc": roc_auc_score(y_true, probability),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))["w4"]
    df = df[df[cfg["active_filter_col"]] == 1].copy()
    group = df["code_module"].astype(str) + "__" + df["code_presentation"].astype(str)
    groups = representation_columns(cfg)
    splitter = GroupKFold(n_splits=5)
    rows = []

    for representation in args.representations:
        if representation not in groups:
            raise ValueError(f"Unknown representation: {representation}")
        columns = groups[representation]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"{representation} missing columns: {missing}")
        x = df[columns]
        y = df["y_dropout"].astype(int)

        for model_name in args.models:
            if representation == "F1_native":
                continue
            for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups=group), start=1):
                y_train = y.iloc[train_index]
                y_test = y.iloc[test_index]
                pipeline = make_pipeline(df, columns, model_name, representation, y_train)
                pipeline.fit(x.iloc[train_index], y_train)
                probability = pipeline.predict_proba(x.iloc[test_index])[:, 1]
                rows.append(
                    {
                        "window": "w4",
                        "representation": representation,
                        "model": model_name,
                        "fold": fold,
                        "n_train": len(train_index),
                        "n_test": len(test_index),
                        **metrics(y_test, probability),
                    }
                )
                print(f"completed {representation}/{model_name}/fold-{fold}", flush=True)

    fold_df = pd.DataFrame(rows)
    mean_df = (
        fold_df.groupby(["window", "representation", "model"], as_index=False)
        .agg(
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            auroc_mean=("auroc", "mean"),
            f1_mean=("f1", "mean"),
            recall_mean=("recall", "mean"),
        )
    )
    fold_df.to_csv(args.output_dir / "groupkfold_fold_results.csv", index=False, encoding="utf-8-sig")
    mean_df.to_csv(args.output_dir / "groupkfold_mean_results.csv", index=False, encoding="utf-8-sig")
    print(mean_df.to_string(index=False))


if __name__ == "__main__":
    main()
