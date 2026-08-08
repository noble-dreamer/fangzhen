[CmdletBinding()]
param(
    [ValidateSet("rotational_tx1", "all_tx")]
    [string]$Mode = "rotational_tx1",

    [switch]$SkipComsol,
    [switch]$EvaluateLabels,
    [switch]$ForcePlan,
    [switch]$ForceOperator,
    [switch]$ResumeIncompleteTraining,
    [switch]$OverwriteTraining,
    [switch]$OverwritePredictions,
    [string[]]$SampleIds = @(),
    [int]$Cores = 0
)

$ErrorActionPreference = "Stop"
if ($ResumeIncompleteTraining -and $OverwriteTraining) {
    throw "Choose only one of -ResumeIncompleteTraining or -OverwriteTraining."
}
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Config = if ($Mode -eq "all_tx") {
    "simple/get_pic/rytov/configs/dataset_a_fullwave_rytov_strict_all_tx.json"
} else {
    "simple/get_pic/rytov/configs/dataset_a_fullwave_rytov.json"
}
$OutputName = if ($Mode -eq "all_tx") { "output_strict_all_tx" } else { "output" }
$Plan = Join-Path $ProjectRoot "simple\get_pic\rytov\$OutputName\training_plan.json"

function Invoke-CondaStep {
    param(
        [Parameter(Mandatory = $true)][string]$Environment,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Title
    )
    Write-Host "`n[$Title]" -ForegroundColor Cyan
    & conda run --no-capture-output -n $Environment python -u @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Title failed with exit code $LASTEXITCODE"
    }
}

Push-Location $ProjectRoot
try {
    Invoke-CondaStep -Environment "get_pic" -Title "Algebraic smoke tests" -Arguments @(
        "simple/get_pic/rytov/smoke_test.py"
    )

    if ($ForcePlan -or -not (Test-Path -LiteralPath $Plan)) {
        $PlanArguments = @(
            "simple/get_pic/rytov/build_training_plan.py",
            "--config", $Config
        )
        if ($ForcePlan) {
            $PlanArguments += "--force"
        }
        Invoke-CondaStep -Environment "get_pic" -Title "Build signed training plan" -Arguments $PlanArguments
    } else {
        Write-Host "`n[Build signed training plan] Reusing $Plan" -ForegroundColor Cyan
    }

    if (-not $SkipComsol) {
        $SolveArguments = @(
            "simple/get_pic/rytov/solve_training_corpus.py",
            "--config", $Config
        )
        if ($ResumeIncompleteTraining) {
            $SolveArguments += "--resume-incomplete"
        }
        if ($OverwriteTraining) {
            $SolveArguments += "--overwrite-existing"
        }
        if ($Cores -gt 0) {
            $SolveArguments += @("--cores", "$Cores")
        }
        Invoke-CondaStep -Environment "comsol" -Title "COMSOL weak-basis training corpus" -Arguments $SolveArguments
    } else {
        Write-Host "`n[COMSOL weak-basis training corpus] Skipped by -SkipComsol" -ForegroundColor Yellow
    }

    $FitArguments = @(
        "simple/get_pic/rytov/fit_operator.py",
        "--config", $Config
    )
    if ($ForceOperator) {
        $FitArguments += "--force"
    }
    Invoke-CondaStep -Environment "get_pic" -Title "Fit frozen full-wave Rytov operator" -Arguments $FitArguments

    Invoke-CondaStep -Environment "get_pic" -Title "Validate frozen operator" -Arguments @(
        "simple/get_pic/rytov/validate_operator.py",
        "--config", $Config
    )

    $RunArguments = @(
        "simple/get_pic/rytov/run_inversion.py",
        "--config", $Config
    )
    if ($EvaluateLabels) {
        $RunArguments += "--evaluate-labels"
    }
    if ($OverwritePredictions) {
        $RunArguments += "--overwrite-existing"
    }
    if ($SampleIds.Count -gt 0) {
        $RunArguments += "--sample-ids"
        $RunArguments += $SampleIds
    }
    Invoke-CondaStep -Environment "get_pic" -Title "Invert formal responses" -Arguments $RunArguments

    $ValidateArguments = @(
        "simple/get_pic/rytov/validate_outputs.py",
        "--config", $Config
    )
    if ($SampleIds.Count -gt 0) {
        $ValidateArguments += "--allow-subset"
    }
    Invoke-CondaStep -Environment "get_pic" -Title "Validate formal outputs" -Arguments $ValidateArguments
} finally {
    Pop-Location
}
