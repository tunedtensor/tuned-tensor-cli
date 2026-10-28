# Windows and PowerShell

The CLI runs natively in Windows PowerShell 5.1 and PowerShell 7. Windows x64
setup includes a private Node.js runtime, installs for the current user without
administrator access, and adds `tt` to the user PATH. It does not change your
PowerShell execution policy or your system Node.js installation.

## Install

Until the first Windows release and WinGet catalog submission are published,
use the npm route below. Merging CLI changes alone does not publish a package.

For a release with Windows assets, download and run
`tuned-tensor-cli-<version>-windows-x64.exe` from the
[GitHub releases](https://github.com/tunedtensor/tuned-tensor-cli/releases).
Open a new terminal and run:

```powershell
tt --version
tt
```

An inspectable PowerShell installer is available in
[`scripts/install.ps1`](../scripts/install.ps1). Download it, inspect it, and
execute it with a policy permitted by your organization. It downloads the
release installer, checks its SHA-256 against the release checksum, and runs
setup. `-Version 0.27.0` (example) or `$env:TT_VERSION` pins an exact release;
the default is the latest stable release. It fails if that release has no
Windows asset. The standalone installer currently supports x64 only.

After `TunedTensor.CLI` is accepted into the public WinGet catalog:

```powershell
winget install --id TunedTensor.CLI --exact --source winget
# Open a new PowerShell window after installation.
tt --version
winget upgrade --id TunedTensor.CLI --exact --source winget
winget uninstall --id TunedTensor.CLI --exact
```

The first catalog submission requires a published stable installer; these
commands are not available just because this repository contains manifests.
Setup can also be updated by running a newer installer, and removed through
Windows **Installed apps**. Close running TT sessions before updating.
`tt upgrade` detects setup-managed installs and shows the appropriate instructions.
Uninstall removes the application's PATH entry and managed files, preserving
`~/.tuned-tensor` and project data.

## npm installation, including Windows ARM64

With Node.js 22.19.0 or newer already installed:

```powershell
npm.cmd install -g --ignore-scripts @tuned-tensor/cli
tt.cmd --version
tt.cmd
tt.cmd upgrade
```

You can install Node.js through WinGet first:

```powershell
winget install --id OpenJS.NodeJS.LTS --exact --source winget
# Open a new terminal so Node and npm are on PATH, then run the npm commands above.
```

Use `npm.cmd` and `tt.cmd` explicitly when PowerShell blocks npm's generated
`.ps1` launchers. No execution-policy change is needed. The standalone setup
ships only `tt.cmd`, so plain `tt` works there even with Restricted policy.
`TT` also works because Windows command lookup is case-insensitive.

For npm installs, uninstall with `npm.cmd uninstall -g @tuned-tensor/cli`.
If you selected a custom npm `--prefix`, supply the same prefix on uninstall
and add that directory itself (not a `bin` subdirectory) to your user PATH.
`tt upgrade` finds the prefix of the running installation automatically.
Use `Get-Command tt -All` to find older installations if the wrong version runs.

## Native CLI scope

Account commands, the conversational interface, hardware discovery, spec
inspection, data commands, and pipeline previews run from PowerShell. Environment
variables use PowerShell syntax, for example `$env:TT_NO_UPDATE_CHECK = '1'`.
Use Windows paths and quotes around paths containing spaces.

Some existing runtime features have narrower platform requirements:

- Agent-approved spec creation/editing requires Linux filesystem handle support.
  On Windows, create a spec with `tt init` and edit it in your editor;
  inspect it with `tt spec view` and preview with `tt pipeline run --dry-run`.
- Native Windows x64 decision training is not yet end-to-end certified; Windows
  ARM64 lacks compatible wheels in the current Python lockfiles. The packaged
  LLM training/serving stack still requires a supported Linux/CUDA environment.
- AWS GPU transport uses local rsync and POSIX path conventions. Run that
  workflow, including background training and cancellation, inside WSL2/Linux.
  A native Windows install does not provide those tools or certify that workflow.

See the platform table in the [README](../README.md) and the
[AWS guide](local-runtime/aws-gpu.md) for training prerequisites.

## Release and WinGet maintenance

`scripts/build-windows.ps1` builds the CLI, packs and installs the npm artifact
with `--ignore-scripts`, includes the build's Node 22 x64 runtime and its license,
and compiles an Inno Setup 6 installer. Run it on Windows with Node/npm and
Inno Setup installed. Outputs are under `windows-dist/`.

The Windows CI job runs on PRs and gates npm releases. It tests the built
installer in PowerShell 5.1 and 7: repeated install, private runtime without
system Node on PATH, CLI inspection and dry-run, upgrade ownership, and uninstall
with unrelated PATH entries and user data preserved. It also exercises npm
global installation. This is CLI/distribution coverage, not a real model-training
certification.

The release workflow attaches the tested `.exe`, `.exe.sha256`, and, for stable
versions, `winget-manifests.zip` to the same GitHub release after npm publication.
The manifests contain the actual installer hash and versioned release URL.
Prereleases never generate stable WinGet manifests.

For each stable release:

1. Download and extract `winget-manifests.zip` from the release.
2. Run `winget validate --manifest .\manifests\t\TunedTensor\CLI\<version>`.
3. Follow Microsoft's [sandbox test and submission procedure](https://learn.microsoft.com/en-us/windows/package-manager/package/repository)
   to test the published installer and submit that directory to `microsoft/winget-pkgs`.
4. Wait for the catalog PR to be accepted before announcing WinGet availability.

No catalog PR is opened by the release workflow. Keep the installer AppId and
manifest ProductCode stable so upgrades and uninstall identify the same app.
Release assets must remain immutable; regenerate checksums/manifests after any
signing step if signing is added to the release process.
