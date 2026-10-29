#Requires -Version 5.1
[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$Installer)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = Split-Path $PSScriptRoot -Parent
$version = (Get-Content (Join-Path $repo 'package.json') -Raw | ConvertFrom-Json).version
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('tt smoke ' + [guid]::NewGuid())
$installDir = Join-Path $temporary 'Tuned Tensor'
$originalPath = $env:Path
$shellExecutable = (Get-Process -Id $PID).Path
$originalHome = $env:TUNED_TENSOR_HOME
$originalUpdateCheck = $env:TT_NO_UPDATE_CHECK
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
New-Item -ItemType Directory -Force $temporary | Out-Null
$env:TUNED_TENSOR_HOME = Join-Path $temporary 'user state'
$env:TT_NO_UPDATE_CHECK = '1'
New-Item -ItemType Directory $env:TUNED_TENSOR_HOME | Out-Null
Set-Content (Join-Path $env:TUNED_TENSOR_HOME 'keep.txt') 'user data'
$uninstaller = Join-Path $installDir 'unins000.exe'
Push-Location $temporary
try {
    foreach ($case in @('node_modules', 'ordinary files', 'wrong marker')) {
        $collision = Join-Path $temporary $case
        New-Item -ItemType Directory $collision -Force | Out-Null
        $keep = Join-Path $collision 'keep.txt'
        if ($case -ne 'ordinary files') {
            New-Item -ItemType Directory (Join-Path $collision 'node_modules') | Out-Null
            $keep = Join-Path $collision 'node_modules\keep.txt'
        }
        Set-Content $keep 'unrelated project'
        if ($case -eq 'wrong marker') {
            '{"manager":"another-app"}' | Set-Content (Join-Path $collision 'tt-install.json') -Encoding Ascii
        }
        $refused = Start-Process -FilePath $Installer -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', "/DIR=`"$collision`"" -WindowStyle Hidden -Wait -PassThru
        if ($refused.ExitCode -eq 0 -or (Get-Content $keep) -ne 'unrelated project') {
            throw "Setup must refuse unrelated files ($case)."
        }
    }
    foreach ($attempt in 1..2) {
        $setup = Start-Process -FilePath $Installer -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', "/DIR=`"$installDir`"" -WindowStyle Hidden -Wait -PassThru
        if ($setup.ExitCode -ne 0) { throw "Setup attempt $attempt failed: $($setup.ExitCode)" }
        if (Test-Path (Join-Path $installDir 'tt.ps1')) { throw 'A .ps1 shim would shadow tt.cmd under Restricted execution policy.' }
        $updatedPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        if (@($updatedPath.Split(';') | Where-Object { $_ -eq $installDir }).Count -ne 1) { throw 'Setup must add PATH exactly once.' }
        # Exercise the bundled runtime with neither Node nor npm on PATH.
        $env:Path = "$installDir;$env:WINDIR\System32"
        $actual = & tt --version
        if ($LASTEXITCODE -ne 0 -or $actual.Trim() -ne $version) { throw "Unexpected version: $actual" }
        $restrictedVersion = & $shellExecutable -NoProfile -ExecutionPolicy Restricted -Command 'tt --version'
        if ($LASTEXITCODE -ne 0 -or $restrictedVersion.Trim() -ne $version) { throw 'tt must work under Restricted execution policy.' }
        & TT --help | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Case-insensitive command lookup failed.' }
        $status = & tt status --json
        if ($LASTEXITCODE -ne 0) { throw 'Local status failed.' }
        $status | ConvertFrom-Json | Out-Null
        Copy-Item (Join-Path $repo 'examples\local-runtime\smoke-spec.json') (Join-Path $temporary 'tunedtensor.json') -Force
        & tt spec view '.\tunedtensor.json' | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Spec inspection failed.' }
        $preview = & tt pipeline run --dry-run --json
        if ($LASTEXITCODE -ne 0) { throw 'Pipeline preview failed.' }
        $preview | ConvertFrom-Json | Out-Null
        # Refuse npm mutation of the installer-owned package, without network.
        $upgrade = & tt upgrade --json 2>&1 | Out-String
        if ($LASTEXITCODE -eq 0 -or $upgrade -notmatch 'winget upgrade') { throw 'Installer upgrade guidance is missing.' }
    }
    # User PATH edits made after installation must survive uninstall.
    $sentinel = Join-Path $temporary 'another app'
    [Environment]::SetEnvironmentVariable('Path', "$updatedPath;$sentinel", 'User')
    $remove = Start-Process -FilePath $uninstaller -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART' -WindowStyle Hidden -Wait -PassThru
    if ($remove.ExitCode -ne 0) { throw 'Uninstall failed.' }
    $remainingPath = [Environment]::GetEnvironmentVariable('Path', 'User').Split(';')
    if ($remainingPath -contains $installDir -or $remainingPath -notcontains $sentinel) { throw 'Uninstall changed unrelated PATH entries.' }
    if (Test-Path (Join-Path $installDir 'tt.cmd')) { throw 'Uninstall left the CLI launcher behind.' }
    if ((Get-Content (Join-Path $env:TUNED_TENSOR_HOME 'keep.txt')) -ne 'user data') { throw 'Uninstall removed user state.' }
} finally {
    $env:Path = $originalPath
    $env:TUNED_TENSOR_HOME = $originalHome
    $env:TT_NO_UPDATE_CHECK = $originalUpdateCheck
    if (Test-Path $uninstaller) {
        Start-Process -FilePath $uninstaller -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART' -WindowStyle Hidden -Wait | Out-Null
    }
    [Environment]::SetEnvironmentVariable('Path', $userPath, 'User')
    Pop-Location
    # Windows PowerShell 5.1 needs extended paths for nested npm dependencies.
    $cleanup = [IO.Path]::GetFullPath($temporary)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $cleanup.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Cleanup escaped the temporary directory.' }
    if ($cleanup.StartsWith('\\')) { $cleanup = '\\?\UNC\' + $cleanup.Substring(2) }
    else { $cleanup = '\\?\' + $cleanup }
    Remove-Item -LiteralPath $cleanup -Recurse -Force
}
Write-Host "Windows installer smoke passed in PowerShell $($PSVersionTable.PSVersion)."
# The negative upgrade test deliberately leaves LASTEXITCODE nonzero. Report the
# suite result explicitly so CI does not mistake that expected refusal for failure.
exit 0
