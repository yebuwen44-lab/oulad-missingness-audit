# Auditing Missingness as Predictive Structure

> **Package version: 1.2.0.** This package provides the processed analytical data, frozen numerical results, source code, configuration files, and integrity-verification utilities associated with the manuscript.

Reproducibility package for the manuscript:

**Auditing Missingness as Predictive Structure: A Layered Representation Framework for Early Dropout Warning under Presentation Heterogeneity**

Authors: Bowen Ye, Dengfeng Yao, Hongzhe Liu, and Cheng Xu.

## Scope and claim boundary

This package supports the Week-4 early-dropout analyses reported in the manuscript. It separates:

- F1: observed features with KNN imputation;
- F1_median: the same observed features with median imputation;
- F2: observed features plus static missing indicators;
- F3: F2 plus event-level missingness patterns;
- F4 variants: selective structural, behavioral, and contrast summaries;
- native-NaN tree comparators;
- five-fold GroupKFold and leave-one-presentation-out audits.

The study does **not** claim a universal predictive advantage from missingness features. The transferable contribution is the layered representation-and-audit protocol and the explicit documentation of presentation heterogeneity.

## Data

The processed table in `data/processed/` contains 32,593 learner-course-presentation instances and 121 columns. The main model-fitting cohort contains 27,538 instances that are active-eligible at Week 4; `active_at_w4` is used as a cohort filter and not as a predictor in the main representations.

The table is derived from the Open University Learning Analytics Dataset (OULAD), an anonymised public dataset released under CC BY 4.0. Please cite:

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data, 4*, 170171. https://doi.org/10.1038/sdata.2017.171

The raw OULAD CSV tables are not duplicated here. The processed table is included to permit direct reproduction of the reported modeling analyses.

## Directory structure

```text
config/                  Frozen feature-group definitions
data/processed/          Processed analytical table
figures/                 Frozen manuscript and supplementary figures
results/frozen/          Archived result tables used in the manuscript
results/rerun/           Destination for newly generated rerun outputs
src/                     Portable verification and reproduction scripts
docs/                    Release and licensing notes
```

## Quick start

### 1. Fast integrity verification

Windows: double-click `RUN_00_VERIFY_PACKAGE.bat`.

Windows PowerShell command line:

```powershell
powershell -ExecutionPolicy Bypass -File .\src\01_verify_frozen_assets.ps1
```

Python alternative:

```bash
python src/01_verify_frozen_assets.py
```

The Windows batch file and PowerShell verifier do not require Python. Both implementations check the processed-table dimensions, prevalence, Week-4 cohort, group count, feature counts, frozen AUPRC values, reconstructed LOGO macro values, and SHA-256 hashes.

### 2. Install the analysis environment

Windows: double-click `RUN_01_CREATE_ENVIRONMENT.bat`.

Or install manually:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -r requirements.txt
```

On Linux/macOS, use `.venv/bin/python` instead.

### 3. Re-run the five-fold main analysis

Windows: double-click `RUN_02_GROUPKFOLD_MAIN.bat`.

```bash
python src/02_run_groupkfold_main.py
```

The full default run fits Logistic Regression, LightGBM, and XGBoost across the main representation hierarchy and may take substantial time. Results are written to `results/rerun/groupkfold/`.

### 4. Re-run the 22-group LOGO audit

Windows: double-click `RUN_03_LOGO_AUDIT.bat`.

```bash
python src/03_run_logo_audit.py
```

This is the most computationally expensive workflow. It produces presentation-level AUPRC/AUROC, macro summaries, top-10% and top-20% ranking diagnostics, and paired F3 contrasts against F1_median, F1_native, and F2.

### 5. Re-run interpretability and descriptive analyses

Windows: double-click `RUN_04_INTERPRETABILITY.bat`.

The SHAP workflow follows the archived protocol: the first GroupKFold split is used for the global LightGBM explanation and is treated as an interpretive, not confirmatory, analysis.

## Frozen versus rerun results

- `results/frozen/` contains the values used in the manuscript.
- `results/rerun/` is created automatically when an analysis script is executed and stores newly generated outputs. It is not present in the initial repository because Git does not track empty directories.
- Small numerical differences may occur across operating systems, CPU libraries, and package builds. The frozen values remain the manuscript record.

## Reproducibility status

- Processed analytical table: included.
- Feature configurations: included.
- Five-fold fold-level and mean results: included.
- F4 ablations, proxy audit, SHAP and descriptive summaries: included.
- Supplementary LOGO presentation table and top-k table: included.
- Portable rerun scripts: included.
- Raw OULAD tables: obtain from the official OULAD source if raw-pipeline reconstruction is required.

## Licensing

This repository uses separate licenses for software and data:

- Source code, batch files, PowerShell scripts, configuration files, and associated software documentation are released under the **MIT License**; see `LICENSE`.
- Processed analytical data under `data/processed/` and machine-readable frozen result tables under `results/frozen/` are released under **CC BY 4.0**; see `DATA_LICENSE.md`.
- The source OULAD dataset is also released under CC BY 4.0. Reuse must retain appropriate attribution to the original dataset creators and cite the dataset paper.
- The manuscript text and publication-ready figures are not covered by these repository licenses unless explicitly stated otherwise.

Additional scope and attribution notes are provided in `docs/LICENSES_AND_RELEASE_NOTES.md`.

## Repository

Project repository: https://github.com/yebuwen44-lab/oulad-missingness-audit

The package has completed independent technical review and is prepared for public release. A versioned GitHub release will be archived through Zenodo, and the persistent DOI will then be added to the repository and manuscript.

## Contact

Corresponding author: Dengfeng Yao, Beijing Union University (`tjtdengfeng@buu.edu.cn`).
