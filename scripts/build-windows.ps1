#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\windows-dist'),
    [string]$CompilerPath = (Join-Path $env:ProgramFiles 'Inno Setup 7\ISCC.exe')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') { throw 'Build the Windows installer on Windows x64.' }
$repo = Split-Path $PSScriptRoot -Parent
$manifest = Get-Content (Join-Path $repo 'package.json') -Raw | ConvertFrom-Json
$version = $manifest.version
if ($version -notmatch '^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$') { throw 'Invalid package version.' }
$node = (Get-Command node.exe -ErrorAction Stop).Source
if ((& $node -p 'process.arch') -ne 'x64') { throw 'An x64 Node.js runtime is required.' }
$compiler = $CompilerPath
if (-not (Test-Path $compiler)) { throw 'Install Inno Setup 7 or provide -CompilerPath before building.' }
if ([Diagnostics.FileVersionInfo]::GetVersionInfo($compiler).FileMajorPart -lt 7) {
    throw 'Inno Setup 7 is required for dependency paths longer than 260 characters.'
}
$output = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force $output | Out-Null
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('tt-windows-' + [guid]::NewGuid())
$payload = Join-Path $temporary 'payload'
New-Item -ItemType Directory -Force $payload | Out-Null
Push-Location $repo
try {
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'CLI build failed.' }
    $pack = & npm.cmd pack --json --pack-destination $temporary | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'npm pack failed.' }
    & npm.cmd install -g --ignore-scripts --no-audit --no-fund --prefix $payload (Join-Path $temporary $pack[0].filename)
    if ($LASTEXITCODE -ne 0) { throw 'Installing the packed CLI failed.' }
    # Keep Node private: putting node.exe on PATH would shadow the user's Node.
    New-Item -ItemType Directory (Join-Path $payload 'runtime') | Out-Null
    Copy-Item $node (Join-Path $payload 'runtime\node.exe')
    $nodeLicense = Join-Path (Split-Path $node -Parent) 'LICENSE'
    if (-not (Test-Path $nodeLicense)) { throw 'The Node distribution LICENSE is required.' }
    Copy-Item $nodeLicense (Join-Path $payload 'runtime\LICENSE')
    Copy-Item (Join-Path $repo 'LICENSE') $payload
    # PowerShell resolves .cmd without changing execution policy. Do not ship
    # npm's .ps1 shim, which takes precedence and is blocked under Restricted.
    Remove-Item (Join-Path $payload 'tt'), (Join-Path $payload 'tt.ps1') -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $repo 'packaging\windows\tt.cmd') $payload
    '{"manager":"windows-installer"}' | Set-Content (Join-Path $payload 'tt-install.json') -Encoding Ascii
    & $compiler "/DVersion=$version" "/DPayload=$payload" "/DOutput=$output" (Join-Path $repo 'packaging\windows\installer.iss')
    if ($LASTEXITCODE -ne 0) { throw 'Installer compilation failed.' }
    $installer = Join-Path $output "tuned-tensor-cli-$version-windows-x64.exe"
    $hash = (Get-FileHash $installer -Algorithm SHA256).Hash
    "$hash  $([IO.Path]::GetFileName($installer))" | Set-Content "$installer.sha256" -Encoding Ascii
    & $node (Join-Path $PSScriptRoot 'generate-winget.mjs') $installer $output
    if ($LASTEXITCODE -ne 0) { throw 'WinGet manifest generation failed.' }
} finally {
    Pop-Location
    Remove-Item $temporary -Recurse -Force
}
