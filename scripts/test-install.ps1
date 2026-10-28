#Requires -Version 5.1
# Exercise the downloader without a network request, real setup, or registry edits.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$fixtureDirectory = Join-Path ([IO.Path]::GetTempPath()) ('tt-download-' + [guid]::NewGuid())
New-Item -ItemType Directory $fixtureDirectory | Out-Null
$originalPath = $env:Path
$originalArchitecture = $env:PROCESSOR_ARCHITECTURE
$originalArchitectureW6432 = $env:PROCESSOR_ARCHITEW6432
try {
    # Check unsupported hosts before mocking x64 downloader cases. No network
    # or setup is reached on either unsupported architecture.
    foreach ($architecture in @('ARM64', 'x86')) {
        $env:PROCESSOR_ARCHITECTURE = $architecture
        $env:PROCESSOR_ARCHITEW6432 = $architecture
        $failure = $null
        try { & (Join-Path $PSScriptRoot 'install.ps1') -Version '1.2.3' }
        catch { $failure = $_.Exception.Message }
        if (-not $failure -or $failure -notmatch 'standalone installer requires Windows x64') {
            throw "Unsupported architecture $architecture was not refused: $failure"
        }
    }
    $env:PROCESSOR_ARCHITECTURE = 'AMD64'
    $env:PROCESSOR_ARCHITEW6432 = 'AMD64'
    '@echo off', 'echo 1.2.3', 'exit /b 0' | Set-Content (Join-Path $fixtureDirectory 'tt.cmd') -Encoding Ascii
    $payload = Join-Path $fixtureDirectory 'fixture.exe'
    [IO.File]::WriteAllText($payload, 'fixture installer')
    $hash = (Get-FileHash $payload -Algorithm SHA256).Hash

    function Invoke-Case([string]$Case, [string]$ExpectedError) {
        $testState = @{ setupCalled = $false; requestedUrl = '' }
        function Invoke-RestMethod {
            param($Uri, $Headers)
            $testState.requestedUrl = $Uri
            $assets = @(
                @{ name = 'tuned-tensor-cli-1.2.3-windows-x64.exe'; browser_download_url = 'https://example.test/setup' },
                @{ name = 'tuned-tensor-cli-1.2.3-windows-x64.exe.sha256'; browser_download_url = 'https://example.test/hash' }
            )
            if ($Case -eq 'missing') { $assets = @() }
            return @{ tag_name = 'v1.2.3'; assets = $assets }
        }
        function Invoke-WebRequest {
            param([switch]$UseBasicParsing, $Uri, $OutFile)
            if ($Uri -eq 'https://example.test/setup') {
                Copy-Item $payload $OutFile
            } else {
                $value = $hash
                if ($Case -eq 'checksum') { $value = '0' * 64 }
                "$value  tuned-tensor-cli-1.2.3-windows-x64.exe" | Set-Content $OutFile -Encoding Ascii
            }
        }
        function Start-Process {
            param($FilePath, $ArgumentList, $WindowStyle, [switch]$Wait, [switch]$PassThru)
            $testState.setupCalled = $true
            if ($ArgumentList -notcontains '/VERYSILENT') { throw 'Setup must be unattended.' }
            if ($Case -eq 'exit') { return @{ ExitCode = 7 } }
            return @{ ExitCode = 0 }
        }
        function Get-ItemProperty {
            param($Path, $Name)
            return @{ InstallLocation = $fixtureDirectory }
        }
        $failure = $null
        try { & (Join-Path $PSScriptRoot 'install.ps1') -Version '1.2.3' }
        catch { $failure = $_.Exception.Message }
        if ($ExpectedError) {
            if (-not $failure -or $failure -notmatch $ExpectedError) { throw "Case $Case expected '$ExpectedError', got '$failure'." }
        } elseif ($failure) { throw "Case $Case failed: $failure" }
        if ($testState.requestedUrl -notmatch '/tags/v1.2.3$') { throw 'Pinned version was not requested.' }
        if ($Case -in @('missing', 'checksum') -and $testState.setupCalled) { throw 'Unverified setup must never execute.' }
        if ($Case -eq 'success' -and -not $testState.setupCalled) { throw 'Verified setup did not execute.' }
    }

    Invoke-Case 'missing' 'does not have a Windows installer'
    Invoke-Case 'checksum' 'checksum mismatch'
    Invoke-Case 'exit' 'code 7'
    Invoke-Case 'success' ''
} finally {
    $env:Path = $originalPath
    $env:PROCESSOR_ARCHITECTURE = $originalArchitecture
    $env:PROCESSOR_ARCHITEW6432 = $originalArchitectureW6432
    Remove-Item $fixtureDirectory -Recurse -Force
}
Write-Host 'PowerShell download installer checks passed.'
