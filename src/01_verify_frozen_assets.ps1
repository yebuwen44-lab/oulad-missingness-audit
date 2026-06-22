$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Data = Join-Path $Root "data\processed\oulad_week4_analysis_table.csv"
$Config = Join-Path $Root "config\feature_sets_week4.json"
$Frozen = Join-Path $Root "frozen_values.json"
$Mean = Join-Path $Root "results\frozen\groupkfold\enhanced_cv_mean_results_v2.csv"
$Xgb = Join-Path $Root "results\frozen\groupkfold\xgboost_robustness_mean_results_w4_v1.csv"
$Logo = Join-Path $Root "results\frozen\logo\logo_lightgbm_presentation_results.csv"
$Hashes = Join-Path $Root "FROZEN_SHA256SUMS.txt"

function Fail([string]$Message) {
    Write-Host "[FAIL] $Message" -ForegroundColor Red
    exit 1
}

function Close-Enough([double]$A, [double]$B, [double]$Tolerance) {
    return ([Math]::Abs($A - $B) -le $Tolerance)
}

try {
    foreach ($Path in @($Data, $Config, $Frozen, $Mean, $Xgb, $Logo, $Hashes)) {
        if (-not (Test-Path -LiteralPath $Path)) {
            Fail "Missing required file: $Path"
        }
    }

    $Expected = Get-Content -LiteralPath $Frozen -Raw -Encoding UTF8 | ConvertFrom-Json

    # Dataset audit. Import-Csv is available in Windows PowerShell and requires no Python installation.
    $Rows = @(Import-Csv -LiteralPath $Data)
    if ($Rows.Count -eq 0) { Fail "Processed table is empty." }

    $Columns = @($Rows[0].PSObject.Properties.Name)
    foreach ($RequiredColumn in @("code_module", "code_presentation", "y_dropout", "active_at_w4")) {
        if ($Columns -notcontains $RequiredColumn) {
            Fail "Processed table is missing column: $RequiredColumn"
        }
    }

    [long]$YSum = 0
    [long]$ActiveN = 0
    [long]$ActiveY = 0
    $Groups = New-Object 'System.Collections.Generic.HashSet[string]'

    foreach ($Row in $Rows) {
        $Y = [int][double]$Row.y_dropout
        $Active = [int][double]$Row.active_at_w4
        $YSum += $Y
        $ActiveN += $Active
        $ActiveY += ($Y * $Active)
        [void]$Groups.Add(($Row.code_module + "||" + $Row.code_presentation))
    }

    $DatasetExpected = $Expected.dataset
    if ($Rows.Count -ne [int]$DatasetExpected.rows) { Fail "Dataset row count mismatch: $($Rows.Count)" }
    if ($Columns.Count -ne [int]$DatasetExpected.columns) { Fail "Dataset column count mismatch: $($Columns.Count)" }
    if (-not (Close-Enough ([double]$YSum / $Rows.Count) ([double]$DatasetExpected.full_dropout_prevalence) 1e-12)) { Fail "Full-cohort dropout prevalence mismatch." }
    if ($ActiveN -ne [int]$DatasetExpected.active_week4_rows) { Fail "Active Week-4 row count mismatch: $ActiveN" }
    if (-not (Close-Enough ([double]$ActiveY / $ActiveN) ([double]$DatasetExpected.active_week4_dropout_prevalence) 1e-12)) { Fail "Active Week-4 dropout prevalence mismatch." }
    if ($Groups.Count -ne [int]$DatasetExpected.module_presentation_groups) { Fail "Module-presentation group count mismatch: $($Groups.Count)" }
    Write-Host "[PASS] Dataset shape, prevalence, active cohort, and group count" -ForegroundColor Green

    # Feature counts.
    $Cfg = (Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json).w4
    $ActualCounts = [ordered]@{
        F1 = @($Cfg.F1).Count
        F1_median = @($Cfg.F0).Count
        F2 = @($Cfg.F2).Count
        F3 = @($Cfg.F3).Count
        F4_all = @($Cfg.F4).Count
        F4_structural_only = @($Cfg.F3).Count + 1
        F4_behavioral_only = @($Cfg.F3).Count + 2
        F4_delta_only = @($Cfg.F3).Count + 1
    }
    foreach ($Property in $Expected.feature_counts_week4.PSObject.Properties) {
        if ([int]$ActualCounts[$Property.Name] -ne [int]$Property.Value) {
            Fail "Feature-count mismatch for $($Property.Name): $($ActualCounts[$Property.Name])"
        }
    }
    Write-Host "[PASS] Week-4 feature-group counts" -ForegroundColor Green

    # Five-fold frozen values.
    $Values = @{}
    foreach ($Row in @(Import-Csv -LiteralPath $Mean)) {
        if ($Row.window -ne "w4") { continue }
        $Model = switch ($Row.model) {
            "lgbm" { "LightGBM" }
            "lr" { "LogisticRegression" }
            default { $null }
        }
        if ($null -ne $Model) {
            $Rep = switch ($Row.feature_set) {
                "F0" { "F1_median" }
                "F4" { "F4_all" }
                default { $Row.feature_set }
            }
            $Values[($Model + "|" + $Rep)] = [double]$Row.auprc_mean
        }
    }
    foreach ($Row in @(Import-Csv -LiteralPath $Xgb)) {
        if ($Row.window -ne "w4") { continue }
        $Rep = switch ($Row.feature_set) {
            "F0" { "F1_median" }
            "F4" { "F4_all" }
            default { $Row.feature_set }
        }
        $Values[("XGBoost|" + $Rep)] = [double]$Row.auprc_mean
    }

    foreach ($ModelProperty in $Expected.five_fold_auprc.PSObject.Properties) {
        $ModelName = $ModelProperty.Name
        foreach ($RepProperty in $ModelProperty.Value.PSObject.Properties) {
            $Key = $ModelName + "|" + $RepProperty.Name
            if (-not $Values.ContainsKey($Key)) { Fail "Missing five-fold value for $Key" }
            if (-not (Close-Enough ([double]$Values[$Key]) ([double]$RepProperty.Value) 5e-7)) {
                Fail ("Five-fold value mismatch for {0}: {1} vs {2}" -f $Key, $Values[$Key], $RepProperty.Value)
            }
        }
    }
    Write-Host "[PASS] Frozen five-fold AUPRC values" -ForegroundColor Green

    # LOGO macro values.
    $LogoRows = @(Import-Csv -LiteralPath $Logo)
    foreach ($Property in $Expected.logo_lightgbm_macro_auprc.PSObject.Properties) {
        [double]$Sum = 0
        foreach ($Row in $LogoRows) { $Sum += [double]$Row.($Property.Name) }
        $MeanValue = $Sum / $LogoRows.Count
        if (-not (Close-Enough $MeanValue ([double]$Property.Value) 1.5e-4)) {
            Fail "LOGO macro mismatch for $($Property.Name): $MeanValue vs $($Property.Value)"
        }
    }
    Write-Host "[PASS] Frozen LightGBM LOGO macro AUPRC values" -ForegroundColor Green

    # Frozen SHA-256 manifest.
    foreach ($Line in Get-Content -LiteralPath $Hashes -Encoding UTF8) {
        $Line = $Line.Trim()
        if ([string]::IsNullOrWhiteSpace($Line)) { continue }
        if ($Line.Length -lt 67) { Fail "Malformed SHA-256 manifest line: $Line" }
        $Digest = $Line.Substring(0, 64).ToLowerInvariant()
        $Relative = $Line.Substring(66)
        $Target = Join-Path $Root $Relative
        if (-not (Test-Path -LiteralPath $Target)) { Fail "Frozen hash target missing: $Relative" }
        $Actual = (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $Digest) { Fail "SHA-256 mismatch: $Relative" }
    }
    Write-Host "[PASS] Frozen SHA-256 manifest" -ForegroundColor Green

    Write-Host ""
    Write-Host "All package integrity checks passed." -ForegroundColor Cyan
    exit 0
}
catch {
    Fail $_.Exception.Message
}
