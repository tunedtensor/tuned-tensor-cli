# Use your AWS GPU with the local pipeline

TT runs one pipeline on your laptop. Add `gpu` to `local-runner.json` to send
adapter training and CUDA evaluation, or foundation pretraining, fine-tuning,
RL and evaluation, to an existing EC2 instance in your account. Foundation
tokenization, dataset preparation, scoring, comparisons, run history, and
artifact validation stay local. Results return to the usual local directories.

```json
{
  "gpu": {
    "provider": "aws",
    "profile": "research",
    "region": "eu-west-1",
    "instanceId": "i-0123456789abcdef0",
    "user": "ubuntu",
    "identityFile": "~/.ssh/research-gpu.pem",
    "maxSeconds": 86400
  }
}
```

The laptop needs AWS CLI v2, OpenSSH, rsync 3.2+, and uv. On macOS, install
a current rsync (for example with Homebrew); the system version is too old. The instance needs Linux,
a compatible NVIDIA GPU/driver, uv, rsync 3.2+, bash, setsid, and GNU timeout. TT
copies the bundled Python source and lockfile and runs that exact runtime
with `uv run --frozen`; there is no separate training image or Step Functions
pipeline. Internet access on the instance is needed to install the locked
runtime. No TT token is needed for training.

`profile` and `region` are optional. AWS CLI uses its normal credential chain,
including environment credentials, configured profiles and SSO. For SSO, log
in first with `aws sso login --profile research`. The selected identity needs
`ec2:DescribeInstances`. TT resolves the instance using
[AWS DescribeInstances](https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instances.html).
An AWS profile authorizes the lookup; SSH access to the machine is also required.

Verify the instance's SSH host key and connect once yourself before using TT.
TT requires a known host key and noninteractive SSH authentication. Omit
`identityFile` to use your SSH agent or SSH configuration. Relative identity
file paths resolve beside the runner config; `~` expands locally. Set
`privateIp: true` when connected to the instance's VPC through a VPN or other
private network. The default uses its public IPv4 address.

```bash
# Adapter workflow: fetch the base model locally first.
tt models prefetch tunedtensor.json --config local-runner.json
tt doctor tunedtensor.json --config local-runner.json
tt pipeline run --spec tunedtensor.json --config local-runner.json

# Same command for a foundation spec; interrupted pretraining can resume.
tt pipeline run --spec tunedtensor.json --config local-runner.json \
  --resume /absolute/path/to/foundation-run
```

Keep pipeline step targets set to `local`: the target describes the
orchestrator, and `gpu` selects where its GPU processes execute. CPU adapter
evaluation (`evaluation.inference.device: "cpu"`) stays on the laptop.
Remove `gpu` to execute everything locally. `--dry-run` or `dryRun: true`
does not connect to AWS or copy files.

TT copies only each process's declared data, model snapshot, and runtime files,
not the workspace or laptop credentials. Adapter runs use an immutable local
model snapshot; the Hugging Face credential/cache root is never uploaded.
GPU processes inherit the remote machine's environment plus a small set of
explicit runtime settings. Gated-model authentication happens during local
prefetch. Output logs, models and foundation checkpoints are downloaded before
temporary remote files are removed. Staging paths are stable for checkpoint
resume and reserved exclusively; an existing directory blocks a new process
until its previous work or artifacts are recovered. Inputs and runtime files are staged for
each process, so large models incur transfer time and require space on both
machines. Serving remains local in this version.

Cancellation terminates the remote process group and retrieves available
outputs. If SSH or artifact retrieval fails, the error includes the remote
staging directory for recovery; keep that directory until you have recovered
your files. A remote timeout (24 hours by default, configurable from 60 seconds
to 7 days per process) limits work after a lost connection or laptop failure.
The pipeline needs the laptop to stay running; it is not a detached scheduler.
Foundation pretraining can resume from returned checkpoints using its existing
`--resume` command. Adapter training keeps its existing restart behavior.

You provide capacity and pay AWS directly. TT does not create, start, stop,
terminate or resize instances, request quotas, choose fallback instances or
fall back to another account. An AWS profile alone cannot create capacity.
The instance remains running after the job; its lifecycle is yours to manage.

Hosted training submission (`tt cloud runs start` and `estimate`) is retired.
Existing cloud reports and cancellation of older hosted jobs remain available.
This CLI change does not delete deployed AWS infrastructure or historical data.
