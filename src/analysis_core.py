#!/usr/bin/env python3
"""Shared preprocessing and model helpers for the OULAD reproduction scripts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def infer_types(df: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    categorical = [c for c in columns if str(df[c].dtype) in {"object", "category", "bool"}]
    numeric = [c for c in columns if c not in categorical]
    return categorical, numeric


def representation_columns(cfg_w4: dict) -> dict[str, list[str]]:
    f3 = list(cfg_w4["F3"])
    return {
        "F1": list(cfg_w4["F1"]),
        "F1_median": list(cfg_w4["F0"]),
        "F2": list(cfg_w4["F2"]),
        "F3": f3,
        "F4_all": list(cfg_w4["F4"]),
        "F4_structural_only": f3 + ["struct_missing_score_w4"],
        "F4_behavioral_only": f3 + ["behavior_missing_score_w4", "has_behavior_missing_w4"],
        "F4_delta_only": f3 + ["behavior_minus_struct_score_w4"],
        "F1_native": list(cfg_w4["F0"]),
    }


def make_preprocessor(
    df: pd.DataFrame,
    columns: list[str],
    model_name: str,
    representation: str,
) -> ColumnTransformer:
    categorical, numeric = infer_types(df, columns)
    is_native = representation == "F1_native"
    is_knn = representation == "F1"

    if is_native:
        numeric_transformer = "passthrough"
    elif is_knn:
        numeric_steps: list[tuple[str, object]] = [("imputer", KNNImputer(n_neighbors=5))]
        if model_name == "lr":
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_transformer = Pipeline(numeric_steps)
    else:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if model_name == "lr":
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_transformer = Pipeline(numeric_steps)

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ]
    )


def make_model(model_name: str, y_train: pd.Series):
    if model_name == "lr":
        return LogisticRegression(
            C=1.0,
            penalty="l2",
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos else 1.0

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            verbose=-1,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            gamma=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def make_pipeline(df: pd.DataFrame, columns: list[str], model_name: str, representation: str, y_train: pd.Series):
    if representation == "F1_native" and model_name == "lr":
        raise ValueError("F1_native is defined only for LightGBM and XGBoost.")
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(df, columns, model_name, representation)),
            ("model", make_model(model_name, y_train)),
        ]
    )
