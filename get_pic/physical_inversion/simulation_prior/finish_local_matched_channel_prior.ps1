param(
    [int]$WaitForPid = 0,
    [string]$Config = 'simple/get_pic/physical_inversion/simulation_prior/configs/dataset_a_channel_prior.json',
    [switch]$RefitPrior
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$conda = (Get-Command conda -ErrorAction Stop).Source
Set-Location $projectRoot

if ($WaitForPid -gt 0) {
    Wait-Process -Id $WaitForPid -ErrorAction SilentlyContinue
}

$configJson = Get-Content -Raw $Config | ConvertFrom-Json
$corpusRoot = Join-Path $projectRoot $configJson.output_root
$responseRoot = Join-Path $corpusRoot 'channel_corpus\frequency_response'
$expectedCount = [int]$configJson.corpus_sample_count
$actualCount = @(Get-ChildItem -LiteralPath $responseRoot -File -Filter '*_H_complex.npz').Count
if ($actualCount -ne $expectedCount) {
    throw "COMSOL corpus is incomplete: expected $expectedCount complete responses, found $actualCount. No prior was fitted."
}

$priorMetadata = Join-Path $corpusRoot 'channel_prior.json'
$priorModel = Join-Path $corpusRoot 'channel_prior.npz'
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
        throw "Channel-prior fitting failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Output "Using existing validated v2 prior: $priorMetadata"
}

& $conda run --no-capture-output -n get_pic python -u `
    simple/get_pic/physical_inversion/run_simulation_channel_prior_inversion.py
if ($LASTEXITCODE -ne 0) {
    throw "Formal channel-prior inversion failed with exit code $LASTEXITCODE"
}
