# Install, upgrade and uninstall

## Windows

On Windows (PowerShell 5.1 or 7), with Node.js 22.19+ installed:

```powershell
winget install --id OpenJS.NodeJS.LTS --exact --source winget # if Node is missing
# Open a new terminal after installing Node.
npm.cmd install -g --ignore-scripts @tuned-tensor/cli
tt.cmd --version
```

The [Windows guide](windows.md) covers native setup with a bundled Node
runtime, PowerShell execution policies, upgrades, and the WinGet distribution
process. `TunedTensor.CLI` will become installable through WinGet after the first
Windows release and catalog submission are accepted.

## Linux and macOS

On Linux or macOS:

```bash
curl -fsSL https://tunedtensor.com/install.sh | sh
tt --version
```

The installer bootstraps Node.js 22.19+ if needed, installs
`@tuned-tensor/cli` with npm `--ignore-scripts`, and falls back to
`~/.local` when the global npm prefix is not writable. Inspect it first
with `curl -fsSL https://tunedtensor.com/install.sh | less`. Pin a
dist-tag or version on `sh`, not `curl`:

```bash
curl -fsSL https://tunedtensor.com/install.sh | TT_VERSION=beta sh
curl -fsSL https://tunedtensor.com/install.sh | TT_VERSION=0.30.0 sh
```

Or install the package directly:

```bash
npm install -g --ignore-scripts @tuned-tensor/cli
tt --version
```

`TT` also works on case-insensitive filesystems, including standard macOS installs.
On case-sensitive systems, use `tt` or add a shell alias (`alias TT=tt`).
Upgrade later with `tt upgrade`. Uninstall with `npm uninstall -g @tuned-tensor/cli` (add
`--prefix ~/.local` if the curl installer used that prefix).

## Requirements

Node.js 22.19.0 or newer is required. Training additionally needs
[`uv`](https://docs.astral.sh/uv/) locally and a supported NVIDIA CUDA host,
either local or [in your AWS account](local-runtime/aws-gpu.md). The laptop
does not need a GPU when `gpu` is configured. The locked Python runner ships
with the npm package and is prepared on first use.

## Run from source

```bash
git clone https://github.com/tunedtensor/tuned-tensor-cli.git
cd tuned-tensor-cli
npm install
npm run build
npm link
```

## Updates

`tt` checks npm for a newer stable release at most every 12 hours and
remembers the answer in `~/.tuned-tensor/update-check.json`. The notice appears
when the shell opens and after commands run in a terminal; explicit commands
never wait on the registry. Upgrade with:

```bash
tt upgrade            # latest stable, into the same npm prefix tt runs from
tt upgrade --check    # only report
tt upgrade beta       # or a specific version
```

Offline or unavailable registry checks are ignored. `CI`, `TT_NO_UPDATE_CHECK=1`
or `NO_UPDATE_NOTIFIER=1` disable the check.

## Compatibility notes

TT includes the former TT Local runtime. `tt local …` and `tt run` remain
hidden compatibility aliases; new scripts should use root commands and
`tt pipeline run --spec tunedtensor.json`.

Earlier releases offered an optional hosted service (`tt cloud`, `tt auth`,
`tt usage`, `tt balance`, `tt topup`, `tt publish`, and managed agent inference
as `tunedtensor/managed`). That service is retired and those commands are
removed. Local projects, run stores and saved provider settings keep working
unchanged; a saved `tunedtensor/managed` selection counts as unconfigured, so
choose your own provider with `/login` and `/model`.
