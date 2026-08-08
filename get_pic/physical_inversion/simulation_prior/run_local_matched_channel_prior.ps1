param(
    [int]$Cores = 16,
    [int]$HeartbeatSeconds = 120,
    [string]$Config = 'simple/get_pic/physical_inversion/simulation_prior/configs/dataset_a_channel_prior.json',
    [string]$OnlySample = '',
    [switch]$SolveOnly,
    [switch]$DryRun,
    [switch]$OverwriteExisting,
    [switch]$RefitPrior
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$conda = (Get-Command conda -ErrorAction Stop).Source
Set-Location $projectRoot

& $conda run --no-capture-output -n comsol python -u `
    simple/get_pic/physical_inversion/simulation_prior/build_channel_corpus_plan.py `
    --config $Config --reuse-compatible
if ($LASTEXITCODE -ne 0) {
    throw "Channel corpus plan failed with exit code $LASTEXITCODE"
}

$solveArguments = @(
    'run', '--no-capture-output', '-n', 'comsol', 'python', '-u',
    'simple/get_pic/physical_inversion/simulation_prior/solve_channel_corpus.py',
    '--config', $Config,
    '--resume-incomplete',
    '--skip-label-preview',
    '--linear-solver', 'pardiso',
    '--cores', $Cores,
    '--heartbeat-s', $HeartbeatSeconds
)
if ($OnlySample) {
    $solveArguments += @('--sample-id', $OnlySample)
}
if ($DryRun) {
    $solveArguments += '--dry-run'
}
if ($OverwriteExisting) {
    $solveArguments += '--overwrite-existing'
}
& $conda @solveArguments
if ($LASTEXITCODE -ne 0) {
    throw "Channel corpus COMSOL solve failed with exit code $LASTEXITCODE"
}

if ($DryRun -or $SolveOnly -or $OnlySample) {
    Write-Output 'Corpus solve stage completed; fitting and formal inversion were intentionally skipped.'
    exit 0
}

$configJson = Get-Content -Raw $Config | ConvertFrom-Json
$priorRoot = Join-Path $projectRoot $configJson.output_root
$priorMetadata = Join-Path $priorRoot 'channel_prior.json'
$priorModel = Join-Path $priorRoot 'channel_prior.npz'
if ($RefitPrior -or -not ((Test-Path -LiteralPath $priorMetadata) -and (Test-Path -LiteralPath $priorModel))) {
    $fitArguments = @(
        'run', '--no-capture-output', '-n', 'get_pic', 'python', '-u',
        'simple/get_pic/physical_inversion/simulation_prior/fit_channel_prior.py',
        '--config', $Config
    )
    if ($RefitPrior) {
        $fitArguments += '--force'
    }
    & $conda @fitArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Channel prior fitting failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Output "Using existing validated v2 prior: $priorMetadata"
}

& $conda run --no-capture-output -n get_pic python -u `
    simple/get_pic/physical_inversion/run_simulation_channel_prior_inversion.py
if ($LASTEXITCODE -ne 0) {
    throw "Formal simulation-prior inversion failed with exit code $LASTEXITCODE"
}
