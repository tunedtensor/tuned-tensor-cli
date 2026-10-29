#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$PackagePath,
    [ValidateSet('Full', 'Install', 'Upgrade', 'Verify', 'Cleanup')][string]$Phase = 'Full',
    [string]$Workspace,
    [ValidateRange(1, 3600)][int]$NpmTimeoutSeconds = 1500
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = Split-Path $PSScriptRoot -Parent
$version = (Get-Content (Join-Path $repo 'package.json') -Raw | ConvertFrom-Json).version
function Assert-TemporaryWorkspace([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $roots = @([IO.Path]::GetTempPath())
    if ($env:RUNNER_TEMP) { $roots += $env:RUNNER_TEMP }
    $inside = $false
    foreach ($root in $roots) {
        $boundary = [IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
        if ($full.StartsWith($boundary, [StringComparison]::OrdinalIgnoreCase)) { $inside = $true }
    }
    if (-not $inside -or [IO.Path]::GetFileName($full) -notlike 'tt npm *') { throw 'Smoke workspace must be a named child of a temporary directory.' }
    return $full
}
function Read-OwnedWorkspace([string]$Path) {
    $full = Assert-TemporaryWorkspace $Path
    if ((Get-Item -LiteralPath $full).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Smoke workspace must not be a reparse point.' }
    $marker = Get-Content -LiteralPath (Join-Path $full '.tt-npm-smoke.json') -Raw | ConvertFrom-Json
    if ($marker.kind -ne 'tt-npm-smoke-v1' -or $marker.root -ne $full) { throw 'Refusing an unowned smoke workspace.' }
    return $marker
}
function Remove-OwnedWorkspace([string]$Path) {
    $full = Assert-TemporaryWorkspace $Path
    if (-not (Test-Path -LiteralPath $full)) { return }
    $null = Read-OwnedWorkspace $full
    $cleanup = if ($full.StartsWith('\\')) { '\\?\UNC\' + $full.Substring(2) } else { '\\?\' + $full }
    Remove-Item -LiteralPath $cleanup -Recurse -Force
}
if ($Phase -ne 'Full' -and -not $Workspace) { throw '-Workspace is required for phased checks.' }
$temporary = if ($Workspace) { Assert-TemporaryWorkspace $Workspace } else { Assert-TemporaryWorkspace (Join-Path ([IO.Path]::GetTempPath()) ('tt npm ' + [guid]::NewGuid())) }
if ($Phase -eq 'Cleanup') { Remove-OwnedWorkspace $temporary; return }
if ($Phase -in @('Full', 'Install')) {
    if ((Test-Path -LiteralPath $temporary) -and ((Get-Item -LiteralPath $temporary).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Smoke workspace must not be a reparse point.' }
    if ((Test-Path -LiteralPath $temporary) -and (Get-ChildItem -LiteralPath $temporary -Force)) { throw 'Refusing a nonempty smoke workspace.' }
    New-Item -ItemType Directory -Force $temporary | Out-Null
    $marker = [PSCustomObject]@{ kind = 'tt-npm-smoke-v1'; root = $temporary; installs = 0 }
    $marker | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $temporary '.tt-npm-smoke.json') -Encoding UTF8
} else {
    $marker = Read-OwnedWorkspace $temporary
    if ($marker.installs -lt 1) { throw 'Install must complete before upgrade or verification.' }
}
$prefix = Join-Path $temporary 'prefix with spaces'
$originalPath = $env:Path
$originalHome = $env:TUNED_TENSOR_HOME
$originalUpdateCheck = $env:TT_NO_UPDATE_CHECK
$env:TUNED_TENSOR_HOME = Join-Path $temporary 'user state'
$env:TT_NO_UPDATE_CHECK = '1'
Push-Location $repo
try {
    if ($Phase -eq 'Full' -and -not $PackagePath) {
        $pack = & npm.cmd pack --json --pack-destination $temporary | ConvertFrom-Json
        if ($LASTEXITCODE -ne 0) { throw 'npm pack failed.' }
        $PackagePath = Join-Path $temporary $pack[0].filename
    }
    if ($Phase -ne 'Verify') {
        if (-not $PackagePath) { throw '-PackagePath is required for installation.' }
        $PackagePath = [IO.Path]::GetFullPath($PackagePath)
        $node = (Get-Command node.exe -ErrorAction Stop).Source
        $npm = Join-Path (Split-Path (Get-Command npm.cmd -ErrorAction Stop).Source -Parent) 'node_modules\npm\bin\npm-cli.js'
        if (-not (Test-Path -LiteralPath $npm)) { throw 'Expected npm beside its Windows launcher.' }
    }
    Set-Location $temporary
    $attempts = if ($Phase -eq 'Verify') { 0 } elseif ($Phase -eq 'Full') { 2 } else { 1 }
    for ($attempt = 0; $attempt -lt $attempts; $attempt++) {
        $label = "npm install $($marker.installs + 1) ($Phase, PowerShell $($PSVersionTable.PSVersion))"
        Write-Host "Starting $label"
        $installArgs = @('install', '-g', '--ignore-scripts', '--no-audit', '--no-fund', '--prefer-offline', '--timing', '--logs-dir', (Join-Path $temporary 'npm-logs'), '--prefix', $prefix)
        $replacementProbe = Join-Path $prefix 'node_modules\@tuned-tensor\cli\.tt-reinstall-probe'
        if ($marker.installs -gt 0) {
            # Every PR has the same package version: require an actual replacement,
            # rather than allowing npm to report an already-current registry version.
            Set-Content -LiteralPath $replacementProbe -Value 'must be replaced' -Encoding Ascii
            $installArgs += '--force'
        }
        & $node (Join-Path $repo 'scripts\run-npm-windows.mjs') --npm-cli $npm --package $PackagePath --timeout-ms ($NpmTimeoutSeconds * 1000) --label $label -- @installArgs
        if ($LASTEXITCODE -ne 0) { throw "npm install attempt $attempt failed." }
        if (Test-Path -LiteralPath $replacementProbe) { throw 'npm did not replace the previous package.' }
        $actual = & (Join-Path $prefix 'tt.cmd') --version
        if ($LASTEXITCODE -ne 0 -or $actual.Trim() -ne $version) { throw "Unexpected version: $actual" }
        $marker.installs += 1
        $marker | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $temporary '.tt-npm-smoke.json') -Encoding UTF8
    }
    Write-Host "Checking installed commands in PowerShell $($PSVersionTable.PSVersion)"
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
    if ($Phase -eq 'Full') { Remove-OwnedWorkspace $temporary }
}
Write-Host "Windows npm smoke passed in PowerShell $($PSVersionTable.PSVersion)."
