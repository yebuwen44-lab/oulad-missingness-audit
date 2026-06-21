from __future__ import annotations

"""
14_descriptive_missingness_analysis_v1.py

用途：
1. 读取增强版主表 oulad_main_table_step4_enhanced_v1.csv
2. 生成 structural missingness（结构性缺失，指由课程流程、资源开放节奏或登记机制决定，本来就可能为空的空白）
   与 behavioral missingness（行为性缺失，指学生本应发生但没有发生的学习行为形成的空白）
   的描述性统计
3. 输出：
   - dropout / non-dropout 群体对比表
   - 不同窗口下的机制缺失发生率表
   - behavioral missingness onset（行为性缺失首次出现周）汇总表
   - 连续缺失长度与不活跃长度汇总表
   - 3 张默认折线/柱状图

这一步的目的：
- 不是再训练新模型
- 而是把“机制论文”的描述性证据补硬
- 直接回答：哪类缺失在什么时间点更像风险信号
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_OUTPUT_DIR = PROJECT_ROOT / "results" / "rerun" / "descriptive"
FIG_OUTPUT_DIR = PROJECT_ROOT / "results" / "rerun" / "figures"

TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def first_behavioral_onset(row: pd.Series) -> str:
    """
    根据当前已有窗口特征，粗粒度估计行为性缺失最早出现在哪个窗口。
    """
    if row.get("has_behavior_missing_w2", 0) == 1:
        return "w2"
    if row.get("has_behavior_missing_w4", 0) == 1:
        return "w4"
    if row.get("has_behavior_missing_w6", 0) == 1:
        return "w6"
    return "none"


def main() -> None:
    input_path = PROCESSED_DIR / "oulad_main_table_step4_enhanced_v1.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入主表：{input_path}")

    df = pd.read_csv(input_path)

    required_cols = [
        "y_dropout",
        "static_admin_missing_cnt",
        "imd_missing_flag",
        "region_missing_flag",
        "education_missing_flag",
        "disability_missing_flag",
        "struct_missing_score_w2",
        "struct_missing_score_w4",
        "struct_missing_score_w6",
        "behavior_missing_score_w2",
        "behavior_missing_score_w4",
        "behavior_missing_score_w6",
        "has_behavior_missing_w2",
        "has_behavior_missing_w4",
        "has_behavior_missing_w6",
        "assess_due_but_missing_cnt_w2",
        "assess_due_but_missing_cnt_w4",
        "assess_due_but_missing_cnt_w6",
        "consecutive_due_missing_cnt_w2",
        "consecutive_due_missing_cnt_w4",
        "consecutive_due_missing_cnt_w6",
        "inactive_gap_days_w2",
        "inactive_gap_days_w4",
        "inactive_gap_days_w6",
        "planned_resource_blank_cnt_w2",
        "planned_resource_blank_cnt_w4",
        "planned_resource_blank_cnt_w6",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少关键字段：{missing}")

    # ------------------------------------------------------------------
    # 1. dropout / non-dropout 群体对比
    # ------------------------------------------------------------------
    group_rows = []
    for label, sub in df.groupby("y_dropout"):
        group_name = "dropout" if int(label) == 1 else "non_dropout"
        row = {
            "group": group_name,
            "n": int(len(sub)),
            "static_admin_missing_cnt_mean": round(float(sub["static_admin_missing_cnt"].mean()), 6),
            "imd_missing_rate": round(safe_rate(sub["imd_missing_flag"]), 6),
            "region_missing_rate": round(safe_rate(sub["region_missing_flag"]), 6),
            "education_missing_rate": round(safe_rate(sub["education_missing_flag"]), 6),
            "disability_missing_rate": round(safe_rate(sub["disability_missing_flag"]), 6),
        }
        for w in ["w2", "w4", "w6"]:
            row[f"struct_missing_score_mean_{w}"] = round(float(sub[f"struct_missing_score_{w}"].mean()), 6)
            row[f"behavior_missing_score_mean_{w}"] = round(float(sub[f"behavior_missing_score_{w}"].mean()), 6)
            row[f"has_behavior_missing_rate_{w}"] = round(safe_rate(sub[f"has_behavior_missing_{w}"]), 6)
            row[f"assess_due_but_missing_cnt_mean_{w}"] = round(float(sub[f"assess_due_but_missing_cnt_{w}"].mean()), 6)
            row[f"consecutive_due_missing_cnt_mean_{w}"] = round(float(sub[f"consecutive_due_missing_cnt_{w}"].mean()), 6)
            row[f"inactive_gap_days_mean_{w}"] = round(float(sub[f"inactive_gap_days_{w}"].mean()), 6)
            row[f"planned_resource_blank_cnt_mean_{w}"] = round(float(sub[f"planned_resource_blank_cnt_{w}"].mean()), 6)
        group_rows.append(row)

    group_df = pd.DataFrame(group_rows)

    # ------------------------------------------------------------------
    # 2. 窗口级机制缺失发生率
    # ------------------------------------------------------------------
    window_rows = []
    for w in ["w2", "w4", "w6"]:
        for label, sub in df.groupby("y_dropout"):
            group_name = "dropout" if int(label) == 1 else "non_dropout"
            window_rows.append(
                {
                    "window": w,
                    "group": group_name,
                    "struct_missing_score_mean": round(float(sub[f"struct_missing_score_{w}"].mean()), 6),
                    "behavior_missing_score_mean": round(float(sub[f"behavior_missing_score_{w}"].mean()), 6),
                    "has_behavior_missing_rate": round(safe_rate(sub[f"has_behavior_missing_{w}"]), 6),
                    "assess_due_but_missing_cnt_mean": round(float(sub[f"assess_due_but_missing_cnt_{w}"].mean()), 6),
                    "inactive_gap_days_mean": round(float(sub[f"inactive_gap_days_{w}"].mean()), 6),
                    "planned_resource_blank_cnt_mean": round(float(sub[f"planned_resource_blank_cnt_{w}"].mean()), 6),
                }
            )
    window_df = pd.DataFrame(window_rows)

    # ------------------------------------------------------------------
    # 3. 行为性缺失首次出现窗口
    # ------------------------------------------------------------------
    onset_df = df.copy()
    onset_df["behavioral_onset_window"] = onset_df.apply(first_behavioral_onset, axis=1)
    onset_summary = (
        onset_df.groupby(["y_dropout", "behavioral_onset_window"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    onset_summary["group"] = onset_summary["y_dropout"].map({0: "non_dropout", 1: "dropout"})
    total_by_group = onset_summary.groupby("group")["count"].transform("sum")
    onset_summary["rate_within_group"] = (onset_summary["count"] / total_by_group).round(6)
    onset_summary = onset_summary[["group", "behavioral_onset_window", "count", "rate_within_group"]]

    # ------------------------------------------------------------------
    # 4. 持续长度统计
    # ------------------------------------------------------------------
    duration_rows = []
    for label, sub in df.groupby("y_dropout"):
        group_name = "dropout" if int(label) == 1 else "non_dropout"
        for w in ["w2", "w4", "w6"]:
            duration_rows.append(
                {
                    "group": group_name,
                    "window": w,
                    "consecutive_due_missing_cnt_mean": round(float(sub[f"consecutive_due_missing_cnt_{w}"].mean()), 6),
                    "consecutive_due_missing_cnt_median": round(float(sub[f"consecutive_due_missing_cnt_{w}"].median()), 6),
                    "inactive_gap_days_mean": round(float(sub[f"inactive_gap_days_{w}"].mean()), 6),
                    "inactive_gap_days_median": round(float(sub[f"inactive_gap_days_{w}"].median()), 6),
                }
            )
    duration_df = pd.DataFrame(duration_rows)

    # ------------------------------------------------------------------
    # 5. 输出 CSV
    # ------------------------------------------------------------------
    group_path = TABLE_OUTPUT_DIR / "descriptive_missingness_group_summary_v1.csv"
    window_path = TABLE_OUTPUT_DIR / "descriptive_missingness_window_summary_v1.csv"
    onset_path = TABLE_OUTPUT_DIR / "behavioral_missingness_onset_summary_v1.csv"
    duration_path = TABLE_OUTPUT_DIR / "missingness_duration_summary_v1.csv"

    group_df.to_csv(group_path, index=False, encoding="utf-8-sig")
    window_df.to_csv(window_path, index=False, encoding="utf-8-sig")
    onset_summary.to_csv(onset_path, index=False, encoding="utf-8-sig")
    duration_df.to_csv(duration_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 6. 输出图
    # ------------------------------------------------------------------
    # 图1：dropout / non-dropout 行为性缺失发生率随窗口变化
    fig1 = plt.figure(figsize=(7, 4.5))
    for group_name in ["dropout", "non_dropout"]:
        sub = window_df[window_df["group"] == group_name].copy()
        x = [2, 4, 6]
        y = sub["has_behavior_missing_rate"].tolist()
        plt.plot(x, y, marker="o", label=group_name)
    plt.xlabel("Week")
    plt.ylabel("Behavioral missingness rate")
    plt.title("Behavioral missingness rate by week")
    plt.legend()
    plt.tight_layout()
    fig1_path = FIG_OUTPUT_DIR / "behavioral_missingness_rate_by_week_v1.png"
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)

    # 图2：结构性得分 vs 行为性得分（按窗口分开）
    fig2 = plt.figure(figsize=(7, 4.5))
    for w in ["w2", "w4", "w6"]:
        sub = window_df[window_df["window"] == w].copy()
        # 每个窗口两组均值取平均，仅作为粗图示
        struct_mean = sub["struct_missing_score_mean"].mean()
        behavior_mean = sub["behavior_missing_score_mean"].mean()
        plt.scatter(struct_mean, behavior_mean, label=w)
    plt.xlabel("Structural missingness score mean")
    plt.ylabel("Behavioral missingness score mean")
    plt.title("Structural vs behavioral mechanism summary")
    plt.legend()
    plt.tight_layout()
    fig2_path = FIG_OUTPUT_DIR / "struct_vs_behavior_mechanism_summary_v1.png"
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)

    # 图3：首次行为性缺失出现窗口分布
    onset_plot = onset_summary.pivot(index="behavioral_onset_window", columns="group", values="rate_within_group").fillna(0)
    fig3 = onset_plot.plot(kind="bar", figsize=(7, 4.5)).get_figure()
    plt.xlabel("First behavioral missingness onset window")
    plt.ylabel("Rate within group")
    plt.title("Behavioral missingness onset distribution")
    plt.tight_layout()
    fig3_path = FIG_OUTPUT_DIR / "behavioral_missingness_onset_distribution_v1.png"
    fig3.savefig(fig3_path, dpi=200)
    plt.close(fig3)

    print("=" * 90)
    print("描述性缺失机制分析完成")
    print(f"群体汇总表: {group_path}")
    print(f"窗口汇总表: {window_path}")
    print(f"首次出现窗口汇总表: {onset_path}")
    print(f"持续长度汇总表: {duration_path}")
    print(f"图1: {fig1_path}")
    print(f"图2: {fig2_path}")
    print(f"图3: {fig3_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
