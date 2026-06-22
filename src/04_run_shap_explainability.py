from __future__ import annotations

"""
13_shap_explainability_v1.py

用途：
1. 读取增强版主表 oulad_week4_analysis_table.csv
2. 读取增强版特征组配置 feature_sets_week4.json
3. 在 Week 4 条件下训练两个解释性主模型：
   - LightGBM + F3
   - LightGBM + F4_structural_only
4. 使用 SHAP（若本地已安装）输出：
   - 全局特征重要性表
   - 全局 SHAP 条形图
   - Top-N 风险个案解释表
5. 若本地未安装 shap，则给出清晰提示

运行前如需安装：
pip install shap
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "config"
TABLE_OUTPUT_DIR = PROJECT_ROOT / "results" / "rerun" / "interpretability"
FIGURE_DIR = PROJECT_ROOT / "results" / "rerun" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def infer_column_types(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    categorical_cols, numeric_cols = [], []
    for c in feature_cols:
        if str(df[c].dtype) in ["object", "category", "bool"]:
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    return categorical_cols, numeric_cols


def build_preprocessor(categorical_cols: list[str], numeric_cols: list[str]):
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    feature_names = []
    feature_names.extend(numeric_cols)

    cat_pipe = preprocessor.named_transformers_["cat"]
    onehot = cat_pipe.named_steps["onehot"]
    cat_names = onehot.get_feature_names_out(categorical_cols).tolist()
    feature_names.extend(cat_names)
    return feature_names


def build_f4_structural_only(config: dict, window: str) -> list[str]:
    f3 = config[window]["F3"]
    f4 = config[window]["F4"]
    structural_cols = [c for c in f4 if c.endswith(f"struct_missing_score_{window}")]
    return list(dict.fromkeys(f3 + structural_cols))


def fit_one_model(
    work_df: pd.DataFrame,
    feature_cols: list[str],
    group_series: pd.Series,
    target_col: str = "y_dropout",
):
    X = work_df[feature_cols].copy()
    y = work_df[target_col].astype(int)

    categorical_cols, numeric_cols = infer_column_types(X, feature_cols)
    preprocessor = build_preprocessor(categorical_cols=categorical_cols, numeric_cols=numeric_cols)

    # 为了稳定和简单，使用第一个 GroupKFold split 作为解释样本划分
    splitter = GroupKFold(n_splits=min(5, group_series.nunique()))
    train_idx, test_idx = next(splitter.split(X, y, groups=group_series))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor, numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    model.fit(X_train_trans, y_train)
    y_prob = model.predict_proba(X_test_trans)[:, 1]

    return {
        "model": model,
        "preprocessor": preprocessor,
        "X_test_raw": X_test,
        "X_test_trans": X_test_trans,
        "y_test": y_test,
        "y_prob": y_prob,
        "feature_names": feature_names,
    }


def save_global_shap_outputs(
    result: dict,
    tag: str,
):
    explainer = shap.TreeExplainer(result["model"])
    shap_values = explainer.shap_values(result["X_test_trans"])

    # 兼容不同 shap 版本返回格式
    if isinstance(shap_values, list):
        shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_matrix = shap_values

    mean_abs = np.abs(shap_matrix).mean(axis=0)
    imp_df = pd.DataFrame(
        {
            "feature": result["feature_names"],
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    imp_path = TABLE_OUTPUT_DIR / f"shap_global_importance_{tag}.csv"
    imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")

    topn = imp_df.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(topn["feature"], topn["mean_abs_shap"])
    plt.xlabel("mean |SHAP value|")
    plt.ylabel("feature")
    plt.tight_layout()
    fig_path = FIGURE_DIR / f"shap_global_bar_{tag}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    # 风险个案：选预测概率最高的前 10 个样本
    top_idx = np.argsort(-result["y_prob"])[:10]
    case_rows = []
    for rank, idx in enumerate(top_idx, start=1):
        row = {
            "rank": rank,
            "pred_prob": float(result["y_prob"][idx]),
            "true_label": int(result["y_test"].iloc[idx]),
        }

        sample_shap = shap_matrix[idx]
        sample_abs = np.abs(sample_shap)
        top_feat_idx = np.argsort(-sample_abs)[:5]

        for j, fi in enumerate(top_feat_idx, start=1):
            row[f"top{j}_feature"] = result["feature_names"][fi]
            row[f"top{j}_shap"] = float(sample_shap[fi])

        case_rows.append(row)

    case_df = pd.DataFrame(case_rows)
    case_path = TABLE_OUTPUT_DIR / f"shap_top_cases_{tag}.csv"
    case_df.to_csv(case_path, index=False, encoding="utf-8-sig")

    return imp_path, fig_path, case_path


def main() -> None:
    if not SHAP_AVAILABLE:
        raise ImportError("未检测到 shap，请先运行：pip install shap")
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("未检测到 lightgbm，请先运行：pip install lightgbm")

    table_path = PROCESSED_DIR / "oulad_week4_analysis_table.csv"
    config_path = CONFIG_DIR / "feature_sets_week4.json"

    if not table_path.exists():
        raise FileNotFoundError(f"未找到增强版主表：{table_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"未找到增强版特征组配置：{config_path}")

    df = pd.read_csv(table_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    window = "w4"
    active_col = config[window]["active_filter_col"]
    work_df = df[df[active_col] == 1].copy()
    group_series = work_df["code_module"].astype(str) + "__" + work_df["code_presentation"].astype(str)

    feature_sets = {
        "w4_f3_lgbm": config[window]["F3"],
        "w4_f4_structural_only_lgbm": build_f4_structural_only(config, window=window),
    }

    summary_rows = []

    for tag, feature_cols in feature_sets.items():
        result = fit_one_model(
            work_df=work_df,
            feature_cols=feature_cols,
            group_series=group_series,
            target_col="y_dropout",
        )
        imp_path, fig_path, case_path = save_global_shap_outputs(result=result, tag=tag)

        summary_rows.append(
            {
                "tag": tag,
                "feature_count": len(feature_cols),
                "global_importance_csv": str(imp_path),
                "global_bar_png": str(fig_path),
                "top_cases_csv": str(case_path),
            }
        )
        print(f"[完成] {tag}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLE_OUTPUT_DIR / "shap_output_manifest_v1.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 90)
    print("SHAP 解释性分析完成")
    print(f"输出清单: {summary_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
