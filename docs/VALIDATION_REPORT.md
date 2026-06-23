# Reproducibility package validation report

## Included and verified

- Processed table: 32,593 rows, 121 columns.
- Active-eligible Week-4 cohort: 27,538 rows; dropout prevalence 0.1855254557.
- Module-presentation groups: 22.
- Frozen feature counts: F1/F1_median 24, F2 29, F3 42, F4_all 46.
- Frozen five-fold point estimates for Logistic Regression, LightGBM, and XGBoost.
- Frozen F4 ablations, proxy audit, SHAP summaries, and descriptive summaries.
- Supplementary LightGBM presentation-level LOGO table and top-k diagnostics.
- SHA-256 manifest for all frozen data, configuration, result, and figure assets.
- Portable Windows one-click launchers and cross-platform Python entry points.

## Tests performed during assembly

1. All Python files passed `py_compile` syntax validation.
2. The GroupKFold and LOGO scripts passed command-line parser smoke tests.
3. The descriptive missingness workflow completed successfully against the included processed table.
4. The fast package verifier passed all dataset, feature-count, frozen-value, LOGO-summary, and SHA-256 checks.
5. Local absolute paths in retained result manifests were replaced with `<PROJECT_ROOT>`.
6. No API keys, passwords, or authentication tokens were detected.

## Computational testing boundary

The complete multi-model five-fold and 22-group LOGO workflows were not re-executed in full during package assembly because they require many boosted-tree fits and are intentionally provided as long-running reproduction workflows. The manuscript values are preserved in `results/frozen/`; reruns write only to `results/rerun/` and do not overwrite the frozen record.

## Release status

- The repository-authored software is released under the MIT License.
- The processed analytical data and machine-readable frozen result tables are released under CC BY 4.0 with attribution to OULAD.
- `CITATION.cff` contains the confirmed authors, ORCID iDs, package version, repository URL, and the Zenodo all-versions DOI.
- The repository is public and versioned releases are archived through Zenodo.
- The all-versions DOI is https://doi.org/10.5281/zenodo.20792632.
- Python bytecode caches have been removed, and a root `.gitignore` prevents regeneration from being committed.
