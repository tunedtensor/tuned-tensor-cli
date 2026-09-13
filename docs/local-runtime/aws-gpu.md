# Use your AWS GPU with the local pipeline

TT runs one pipeline on your laptop. Add `gpu` to `local-runner.json` to send
adapter training and CUDA evaluation, or foundation pretraining, fine-tuning,
RL and evaluation, to an existing EC2 instance in your account. Foundation
tokenization, dataset preparation, adapter scoring/comparisons, run history,
and artifact validation stay local. Results return to the usual local
directories. The laptop does not need an NVIDIA GPU.

## 1. Install TT and create a project

Use CLI 0.16.0 or newer with Node.js 22.19+ on Linux or macOS:

```bash
npm install -g --ignore-scripts @tuned-tensor/cli
tt --version
mkdir my-aws-training && cd my-aws-training
tt init --name "My AWS Adapter" --model Qwen/Qwen3.5-2B
```

Edit the generated `tunedtensor.json`: replace the placeholder system prompt and both placeholder examples with
your desired instructions and real input/output pairs. For a small functional
trial, use the [four-example smoke spec](../../examples/local-runtime/smoke-spec.json)
instead. It checks the workflow, not model quality. An existing project can keep
its spec and pipeline unchanged.

## 2. Prepare your AWS instance

You need two separate kinds of access: an **AWS profile** to find the instance,
and an **SSH key or agent** to execute commands on it. A TT login is unnecessary.
If you do not already have an instance, launch one in your AWS account or ask
your administrator to provide one with the prerequisites below.

Our CLI 0.16.0 smoke test used a `g5.xlarge` (A10G), an AWS Deep Learning Base
GPU AMI with Ubuntu 24.04, and a 120 GiB root disk in `eu-west-1`. This is a
tested small-workload configuration, not a sizing guarantee for other models.
Use a driver compatible with the bundled CUDA runtime; `tt doctor` checks it.

Before launching, check the EC2 GPU quota in your chosen region. For the tested
G-family On-Demand instance, the quota is **Running On-Demand G and VT instances**
(`L-DB2E81BA`), measured in vCPUs; one `g5.xlarge` needs 4 available vCPUs.
If it is too low, follow [AWS's quota increase procedure](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html).
Approval does not guarantee instance availability. TT does not request capacity
or choose a different region for you.

During launch, select your [EC2 SSH key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html),
allow inbound TCP 22 from your laptop's IP (or use your private network), and
record the instance ID, region, SSH username and IP. Follow
[AWS's SSH connection prerequisites](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-to-linux-instance.html).
Instance creation and quota management need separate AWS permissions; TT itself
only needs `ec2:DescribeInstances` for lookup.

## 3. Configure the connection

Save this as `local-runner.json` beside `tunedtensor.json`,
replacing the example values with your own. You can also copy
[`examples/local-runtime/aws-runner.json`](../../examples/local-runtime/aws-runner.json).
All runner paths refer to the laptop; TT translates the paths needed remotely.

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
a current rsync (for example with Homebrew); the system version is too old.
Each machine must find its required programs through `PATH`; on the instance,
this includes noninteractive SSH sessions. The instance needs Linux, a compatible NVIDIA
GPU/driver, uv, rsync 3.2+, bash, setsid, and GNU timeout. TT
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
The selected instance must already be running in the selected region and have
enough GPU memory and free disk for the model, runtime, inputs and outputs.

Verify the instance's SSH host key for the selected IP address and connect
once yourself before using TT. TT requires a known host key and noninteractive
SSH authentication. Omit
`identityFile` to use your SSH agent or SSH configuration. Relative identity
file paths resolve beside the runner config; `~` expands locally. Set
`privateIp: true` when connected to the instance's VPC through a VPN or other
private network. The default uses its public IPv4 address. Security groups and
network routes must allow the laptop to reach that address over SSH.
TT addresses the instance by IP, so any `Host` rules in your SSH configuration
must match that IP. Remote Python commands run through `bash -lc`, which loads
the login profile so a normal uv installation can be found.

After configuring your AWS profile (for SSO, use `aws configure sso` once),
check access from the same terminal that will run TT:

```bash
# SSO profiles only; skip this line for other credential methods.
aws sso login --profile research
aws sts get-caller-identity --profile research
aws ec2 describe-instances --profile research --region eu-west-1 \
  --instance-ids i-0123456789abcdef0 \
  --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress]' --output text
```

Use the returned IP below. First verify its host fingerprint through your AWS
console or administrator and connect interactively to record the verified key.
Then confirm that unattended SSH works:

```bash
chmod 400 ~/.ssh/research-gpu.pem
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -i ~/.ssh/research-gpu.pem ubuntu@YOUR_INSTANCE_IP \
  'bash -lc "command -v uv rsync setsid timeout; nvidia-smi"'
```

If `uv` is missing, install it on that machine using the
[uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
and make it available in the login shell. Install the other listed prerequisites
with that machine's package manager. Run the SSH check again before training.

## 4. Preview and run

Run these commands on your laptop in the project directory. Adapter workflow:

```bash
# Fetch the base model locally first.
tt validate tunedtensor.json
tt models prefetch tunedtensor.json --config local-runner.json
tt doctor tunedtensor.json --config local-runner.json
tt pipeline run --spec tunedtensor.json --config local-runner.json --dry-run
tt pipeline run --spec tunedtensor.json --config local-runner.json
```

For a foundation spec, no base-model prefetch is needed. Start a new run with:

```bash
tt validate tunedtensor.json
tt doctor tunedtensor.json --config local-runner.json
tt pipeline run --spec tunedtensor.json --config local-runner.json \
  --dry-run
tt pipeline run --spec tunedtensor.json --config local-runner.json
```

Choose the command block for your spec's engine. `tt doctor` connects to the
instance and prepares/probes its locked Python/CUDA runtime, which may take
time on first use. It does not start training or reserve GPU capacity.
`tt hardware` reports the laptop's hardware; it does not inspect the AWS
instance. `tt pipeline run` discovers `local-runner.json` beside the selected
spec when `--config` is omitted.

Keep pipeline step targets set to `local`: the target describes the
orchestrator, and `gpu` selects where its GPU processes execute. CPU adapter
evaluation (`evaluation.inference.device: "cpu"`) stays on the laptop.
Remove `gpu` to execute everything locally. `--dry-run` or `dryRun: true`
does not connect to AWS or copy files.

## 5. Inspect results and finish

Wait for a successful completion message. Adapter runs return a run ID and a
local model ID; use the actual IDs from the output:

```bash
tt runs list --config local-runner.json
tt runs report RUN_ID --config local-runner.json
tt models list --config local-runner.json
tt models verify MODEL_ID --config local-runner.json
```

For foundation runs, inspect `report.json` in the run directory printed by the
command. To resume it, pass `--resume /absolute/path/to/foundation-run` to the
same `tt pipeline run` command. Keep the laptop awake and connected until the
command finishes and outputs have returned. Review the report before another run.

Stop or terminate your instance in AWS when finished, according to your own
retention needs. TT does not do this for you, even after cancellation or timeout.

## Transfers and recovery

TT copies only each process's declared data, model snapshot, and runtime files,
not the workspace or laptop credentials. Adapter runs use an immutable local
model snapshot; the Hugging Face credential/cache root is never uploaded.
GPU processes inherit the remote machine's environment plus a small set of
explicit runtime settings. Gated-model authentication happens during local
prefetch. Logs stream to the laptop; models and foundation checkpoints are
downloaded when the process ends, before temporary remote files are removed.
Staging paths are stable for checkpoint resume and reserved exclusively; an
existing directory blocks a new process until its previous work or artifacts
are recovered. Inputs and runtime files are staged for each process, so large
models incur transfer time and require space on both
machines, including the instance's `/tmp`. Serving remains local in this version.

Foundation `checkpoint_backup_dir` is also a local path that TT stages remotely.
Keep it outside the foundation run directory; overlapping input, output,
recovery or backup paths are rejected before transfer.
During pretraining, both rolling checkpoints and this backup are written under
the instance's staging directory. They return to the laptop only after the
process ends or is cancelled; there is no continuous backup to the laptop or
to a separate remote disk. Protect and recover that staging directory if the
connection fails. See [long foundation runs](foundation-long-runs.md) for the
checkpoint contents and resume contract.

Cancellation signals the remote process group and retrieves available
outputs while SSH remains reachable. If SSH or artifact retrieval fails, the
error includes the remote staging directory for recovery; keep it until you have recovered
your files. A remote timeout (24 hours by default, configurable from 60 seconds
to 7 days per process) limits work after a lost connection or laptop failure.
Adapter evaluation also retains `evaluation.timeoutMs` (30 minutes by default),
and individual transfer/setup commands have a 30-minute timeout. Increasing
`gpu.maxSeconds` does not extend those limits.
The pipeline needs the laptop to stay running; it is not a detached scheduler.
Foundation pretraining can resume from returned checkpoints using its existing
`--resume` command. Use the same laptop run directory and unchanged spec/data
so checkpoint identities still match. Adapter training keeps its existing
restart behavior.

You provide capacity and pay AWS directly. TT does not create, start, stop,
terminate or resize instances, request quotas, choose fallback instances or
fall back to another account. An AWS profile alone cannot create capacity.
The instance remains running after the job; its lifecycle is yours to manage.

Hosted training submission (`tt cloud runs start` and `estimate`) is retired.
Existing cloud reports and cancellation of older hosted jobs remain available.
This CLI change does not delete deployed AWS infrastructure or historical data.

## Troubleshooting a first run

| Symptom | What to check |
| --- | --- |
| AWS credentials expired or access denied | Renew your SSO login, check the selected profile/account, and ask for `ec2:DescribeInstances` permission. |
| Instance not found or not running | Confirm the instance ID and region, and start the instance through AWS. |
| GPU quota or capacity error during launch | Check the regional quota and AWS capacity; a TT token or credit top-up will not fix it. |
| SSH timeout or permission denied | Check IP routing, security group TCP 22 access, SSH username, key and agent using the manual SSH command above. |
| Host key verification failed | Verify the host fingerprint through AWS or your administrator; do not disable strict host checking. |
| `uv` missing or CUDA probe fails | Check the remote login-shell PATH and GPU driver; rerun `tt doctor` after correcting the host. |
| Long pause before the GPU process starts | Model inputs are uploaded for every process. Check network speed and free space on both machines, especially remote `/tmp`. |
| Download fails or staging directory already exists | Preserve the remote directory shown in the error and recover its outputs before retrying. Do not remove the only checkpoint copy. |
