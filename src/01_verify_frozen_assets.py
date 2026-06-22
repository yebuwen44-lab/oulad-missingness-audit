#!/usr/bin/env python3
"""Fast integrity and numerical-freeze verification using the Python standard library."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "oulad_week4_analysis_table.csv"
CONFIG = ROOT / "config" / "feature_sets_week4.json"
FROZEN = ROOT / "frozen_values.json"
MEAN = ROOT / "results" / "frozen" / "groupkfold" / "enhanced_cv_mean_results_v2.csv"
XGB = ROOT / "results" / "frozen" / "groupkfold" / "xgboost_robustness_mean_results_w4_v1.csv"
LOGO = ROOT / "results" / "frozen" / "logo" / "logo_lightgbm_presentation_results.csv"
HASHES = ROOT / "FROZEN_SHA256SUMS.txt"


def close(a: float, b: float, tol: float = 5e-7) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def read_dicts(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message: str):
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def main() -> None:
    expected = json.loads(FROZEN.read_text(encoding="utf-8"))
    for path in [DATA, CONFIG, MEAN, XGB, LOGO, HASHES]:
        if not path.exists():
            fail(f"Missing required file: {path.relative_to(ROOT)}")

    # Dataset audit.
    rows = 0
    y_sum = 0
    active_n = 0
    active_y = 0
    groups = set()
    with DATA.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        required = {"code_module", "code_presentation", "y_dropout", "active_at_w4"}
        missing = required.difference(columns)
        if missing:
            fail(f"Processed table is missing columns: {sorted(missing)}")
        for row in reader:
            rows += 1
            y = int(float(row["y_dropout"]))
            active = int(float(row["active_at_w4"]))
            y_sum += y
            active_n += active
            active_y += y * active
            groups.add((row["code_module"], row["code_presentation"]))

    d = expected["dataset"]
    checks = {
        "rows": rows == d["rows"],
        "columns": len(columns) == d["columns"],
        "full dropout prevalence": close(y_sum / rows, d["full_dropout_prevalence"], 1e-12),
        "active Week-4 rows": active_n == d["active_week4_rows"],
        "active Week-4 dropout prevalence": close(active_y / active_n, d["active_week4_dropout_prevalence"], 1e-12),
        "module-presentation groups": len(groups) == d["module_presentation_groups"],
    }
    for name, ok in checks.items():
        if not ok:
            fail(f"Dataset check failed: {name}")
    print("[PASS] Dataset shape, prevalence, active cohort, and group count")

    # Feature counts.
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["w4"]
    actual_counts = {
        "F1": len(cfg["F1"]),
        "F1_median": len(cfg["F0"]),
        "F2": len(cfg["F2"]),
        "F3": len(cfg["F3"]),
        "F4_all": len(cfg["F4"]),
        "F4_structural_only": len(cfg["F3"]) + 1,
        "F4_behavioral_only": len(cfg["F3"]) + 2,
        "F4_delta_only": len(cfg["F3"]) + 1,
    }
    if actual_counts != expected["feature_counts_week4"]:
        fail(f"Feature-count mismatch: {actual_counts}")
    print("[PASS] Week-4 feature-group counts")

    # Five-fold frozen values.
    values = {}
    for row in read_dicts(MEAN):
        if row["window"] != "w4":
            continue
        model = {"lgbm": "LightGBM", "lr": "LogisticRegression"}.get(row["model"])
        if model:
            rep = {"F0": "F1_median", "F4": "F4_all"}.get(row["feature_set"], row["feature_set"])
            values[(model, rep)] = float(row["auprc_mean"])
    for row in read_dicts(XGB):
        if row["window"] == "w4":
            rep = {"F0": "F1_median", "F4": "F4_all"}.get(row["feature_set"], row["feature_set"])
            values[("XGBoost", rep)] = float(row["auprc_mean"])
    for model, reps in expected["five_fold_auprc"].items():
        for rep, target in reps.items():
            found = values.get((model, rep))
            if found is None or not close(found, target, 5e-7):
                fail(f"Five-fold value mismatch for {model}/{rep}: {found} vs {target}")
    print("[PASS] Frozen five-fold AUPRC values")

    # LOGO macro values reconstructed from the supplementary table.
    logo_rows = read_dicts(LOGO)
    for rep, target in expected["logo_lightgbm_macro_auprc"].items():
        mean = sum(float(r[rep]) for r in logo_rows) / len(logo_rows)
        if not close(mean, target, 1.5e-4):
            fail(f"LOGO macro mismatch for {rep}: {mean:.8f} vs {target}")
    print("[PASS] Frozen LightGBM LOGO macro AUPRC values")

    # Frozen file hashes.
    for line in HASHES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel = line.split("  ", 1)
        path = ROOT / rel
        if not path.exists():
            fail(f"Frozen hash target missing: {rel}")
        if sha256(path) != digest:
            fail(f"SHA-256 mismatch: {rel}")
    print("[PASS] Frozen SHA-256 manifest")
    print("\nAll package integrity checks passed.")


if __name__ == "__main__":
    main()
