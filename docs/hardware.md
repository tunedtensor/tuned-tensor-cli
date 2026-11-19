# Hardware and platform support

`tt hardware` always inspects the machine running the CLI. When `runtime.gpu`
is configured, `tt doctor tunedtensor.json` checks the remote AWS runtime
instead, so a laptop without a GPU can still train there. See
[AWS GPU setup](local-runtime/aws-gpu.md).

The startup **What you can do** panel covers specs/datasets, decision training,
LLM fine-tuning, training from scratch, and serving even before a spec exists.
Decision training can use CPU or Apple MPS; the other training paths currently
require CUDA locally or a configured AWS GPU. A missing NVIDIA GPU does not
prevent spec creation or dataset preparation. The banner lists ready workflows on
one line and shows only the status of the rest; `/system` adds reasons and next steps.

Guidance distinguishes **Ready**, **Needs setup**, **Limited resources**,
**Remote GPU**, **Needs verification**, and **Unsupported runtime**. Quick
startup sampling does not install Python dependencies. `tt hardware` explicitly
checks the locked adapter and decision runtimes separately, including MPS
availability; it may install those runtimes on first use. It does not download
models or certify their fit. Old snapshots without a decision probe show
"Needs verification" until refreshed.

| Platform | Decision training | LLM / foundation training | Packaged LLM serving |
| --- | --- | --- | --- |
| Linux x64 / ARM64 | CPU; CUDA when available | NVIDIA CUDA, subject to memory and runtime checks | Linux + NVIDIA CUDA; separate runtime |
| Apple Silicon macOS | CPU / MPS after runtime checks | Remote AWS GPU | Remote Linux/CUDA host |
| Native Windows x64 | CPU path; end-to-end training not yet certified | Not certified natively; check a Linux/WSL2 environment or use AWS | Linux/CUDA host |
| Intel macOS / Windows ARM64 | No compatible wheels in the current lockfiles | Remote AWS GPU | Remote Linux/CUDA host |

The native Windows CI job verifies installation, PowerShell CLI commands,
pipeline previews, hardware discovery and displayed guidance, not model training.
Agent-approved spec creation, editing and history work on native Windows and Linux.
AWS transport/background training should run inside WSL2/Linux. See
[Windows limitations](windows.md#native-cli-scope).
WSL is assessed as its own Linux environment. AMD/Intel GPU
acceleration is not enabled by the packaged CUDA runtime. Decision resource
warnings use conservative advisory thresholds (8 GiB available working memory,
6 GiB free model-cache disk, 4 CPU cores); they are not measured model minima.
Batch size, sequence length, training mode and other running applications matter.
CPU evaluation does not imply that `tt serve` can run on CPU. Agent chat still
needs a model provider; local data commands do not.
