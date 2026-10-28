#Requires -Version 5.1
[CmdletBinding()]
param([string]$PackagePath)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = Split-Path $PSScriptRoot -Parent
$version = (Get-Content (Join-Path $repo 'package.json') -Raw | ConvertFrom-Json).version
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('tt npm ' + [guid]::NewGuid())
$prefix = Join-Path $temporary 'prefix with spaces'
$originalPath = $env:Path
$originalHome = $env:TUNED_TENSOR_HOME
$originalUpdateCheck = $env:TT_NO_UPDATE_CHECK
New-Item -ItemType Directory -Force $temporary | Out-Null
$env:TUNED_TENSOR_HOME = Join-Path $temporary 'user state'
$env:TT_NO_UPDATE_CHECK = '1'
Push-Location $repo
try {
    if (-not $PackagePath) {
        $pack = & npm.cmd pack --json --pack-destination $temporary | ConvertFrom-Json
        if ($LASTEXITCODE -ne 0) { throw 'npm pack failed.' }
        $PackagePath = Join-Path $temporary $pack[0].filename
    }
    $PackagePath = [IO.Path]::GetFullPath($PackagePath)
    Set-Location $temporary
    foreach ($attempt in 1..2) {
        & npm.cmd install -g --ignore-scripts --no-audit --no-fund --prefer-offline --prefix $prefix $PackagePath
        if ($LASTEXITCODE -ne 0) { throw "npm install attempt $attempt failed." }
        $actual = & (Join-Path $prefix 'tt.cmd') --version
        if ($LASTEXITCODE -ne 0 -or $actual.Trim() -ne $version) { throw "Unexpected version: $actual" }
    }
    $env:Path = "$prefix;$env:Path"
    & TT.cmd --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Case-insensitive npm launcher failed.' }
    $shellExecutable = (Get-Process -Id $PID).Path
    $restrictedVersion = & $shellExecutable -NoProfile -ExecutionPolicy Restricted -Command 'tt.cmd --version'
    if ($LASTEXITCODE -ne 0 -or $restrictedVersion.Trim() -ne $version) { throw 'npm .cmd launcher failed under Restricted policy.' }
    $status = & tt.cmd status --json
    if ($LASTEXITCODE -ne 0) { throw 'npm-installed status failed.' }
    $status | ConvertFrom-Json | Out-Null
    Copy-Item (Join-Path $repo 'examples\local-runtime\smoke-spec.json') '.\tunedtensor.json'
    & tt.cmd spec view '.\tunedtensor.json' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'npm-installed Windows spec path failed.' }
    $preview = & tt.cmd pipeline run --dry-run --json
    if ($LASTEXITCODE -ne 0) { throw 'npm-installed pipeline preview failed.' }
    $preview | ConvertFrom-Json | Out-Null
    $hardware = & tt.cmd hardware --quick --json
    if ($LASTEXITCODE -ne 0) { throw 'npm-installed hardware inspection failed.' }
    $report = $hardware | ConvertFrom-Json
    if ($report.inventory.os.arch -ne (& node -p 'process.arch')) { throw 'Unexpected hardware architecture.' }
} finally {
    $env:Path = $originalPath
    $env:TUNED_TENSOR_HOME = $originalHome
    $env:TT_NO_UPDATE_CHECK = $originalUpdateCheck
    Pop-Location
    Remove-Item -LiteralPath $temporary -Recurse -Force
}
Write-Host "Windows npm smoke passed in PowerShell $($PSVersionTable.PSVersion)."
exit 0
