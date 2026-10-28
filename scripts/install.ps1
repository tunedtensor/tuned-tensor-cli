#Requires -Version 5.1
<# Install the published Windows x64 release without admin access or a Node install.
   Download and inspect this script before running it. TT_VERSION can pin a release. #>
[CmdletBinding()]
param([string]$Version = $env:TT_VERSION)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') { throw 'This installer requires Windows.' }
$architecture = $env:PROCESSOR_ARCHITEW6432
if (-not $architecture) { $architecture = $env:PROCESSOR_ARCHITECTURE }
if ($architecture -ne 'AMD64') { throw 'The standalone installer requires Windows x64. On ARM64, install Node.js and use npm.cmd install -g --ignore-scripts @tuned-tensor/cli.' }
if (-not $Version) { $Version = 'latest' }
if ($Version -ne 'latest' -and $Version -notmatch '^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$') {
    throw 'Version must be latest or an exact release version, such as 0.27.0.'
}
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$api = 'https://api.github.com/repos/tunedtensor/tuned-tensor-cli/releases'
if ($Version -eq 'latest') { $url = "$api/latest" } else { $url = "$api/tags/v$Version" }
$release = Invoke-RestMethod -Uri $url -Headers @{ 'User-Agent' = 'TunedTensor-Installer' }
$resolvedVersion = $release.tag_name -replace '^v', ''
if ($resolvedVersion -notmatch '^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$') { throw 'Release has an invalid version.' }
$filename = "tuned-tensor-cli-$resolvedVersion-windows-x64.exe"
$asset = @($release.assets | Where-Object { $_.name -eq $filename })
$checksum = @($release.assets | Where-Object { $_.name -eq "$filename.sha256" })
if ($asset.Count -ne 1 -or $checksum.Count -ne 1) { throw "Release $resolvedVersion does not have a Windows installer yet. See the Windows installation guide for npm installation." }
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('tt-install-' + [guid]::NewGuid())
New-Item -ItemType Directory $temporary | Out-Null
try {
    $installer = Join-Path $temporary $filename
    Invoke-WebRequest -UseBasicParsing $asset[0].browser_download_url -OutFile $installer
    Invoke-WebRequest -UseBasicParsing $checksum[0].browser_download_url -OutFile "$installer.sha256"
    $expected = (Get-Content "$installer.sha256" -Raw).Trim()
    if ($expected -notmatch ('^([A-Fa-f0-9]{64})  ' + [regex]::Escape($filename) + '$')) { throw 'Invalid release checksum file.' }
    if ((Get-FileHash $installer -Algorithm SHA256).Hash -ne $Matches[1]) { throw 'Installer checksum mismatch.' }
    $setup = Start-Process -FilePath $installer -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART' -WindowStyle Hidden -Wait -PassThru
    if ($setup.ExitCode -ne 0) { throw "Installer exited with code $($setup.ExitCode)." }
    $installDir = (Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\TunedTensor.CLI_is1' -Name InstallLocation).InstallLocation
    $env:Path = "$installDir;$env:Path"
    & (Join-Path $installDir 'tt.cmd') --version
    if ($LASTEXITCODE -ne 0) { throw 'Installed CLI verification failed.' }
    Write-Host 'Tuned Tensor is installed. Run tt in this terminal, or open a new PowerShell window.'
} finally {
    Remove-Item $temporary -Recurse -Force
}
