# Long foundation runs

The foundation engine can run recoverable, single-GPU pretraining from `.txt`
files or JSONL shards whose rows contain a string `text` field. Keep training
and validation data in different files or directories.

The commands below also support an [AWS GPU](aws-gpu.md) configured in the
adjacent `local-runner.json`. The CLI and tokenizer still run locally; the
local machine must stay awake and connected throughout the pipeline.

## Spec

The long-run fields live under `foundation` in `tunedtensor.json`:

```json
{
  "corpus_path": "./data/train",
  "validation_path": "./data/validation.jsonl",
  "checkpoint_backup_dir": "/mnt/checkpoint-backup/my-run",
  "depth": 20,
  "vocab_size": 32000,
  "tokenizer_max_chars": 100000000,
  "max_chars": 50000000000,
  "pretrain_steps": 200000,
  "sequence_length": 1024,
  "batch_size": 2,
  "gradient_accumulation_steps": 16,
  "learning_rate": 0.0003,
  "weight_decay": 0.1,
  "warmup_steps": 2000,
  "min_lr_ratio": 0.1,
  "gradient_clip": 1.0,
  "bf16": true,
  "checkpoint_interval_steps": 500,
  "checkpoint_interval_seconds": 1800,
  "keep_checkpoints": 3,
  "log_interval_steps": 10,
  "seed": 1337,
  "finetune_steps": 100,
  "rl_steps": 0,
  "nproc_per_node": 1
}
```

Relative paths are resolved beside the spec. `max_chars` is a deterministic
upper bound across the sorted corpus shards. Omit `corpus_path` only for the
small prompt/example smoke corpus.

Run the preflight, then use one stable, absolute run directory from the first
launch onward:

```bash
tt doctor tunedtensor.json
tt pipeline run \
  --spec /absolute/project/tunedtensor.json \
  --resume /absolute/project/.tuned-tensor/foundation-runs/experiment-1
```

`--resume` also starts a missing directory. On restart, verified completed
stages are skipped. An interrupted pretrain stage restores the newest complete
checkpoint with matching configuration. Changing model, optimizer, tokenizer,
data manifest, or schedule settings requires a new run directory.

Recovery data is stored under `pretrain/recovery/`:

- `pretrain-tokens.bin` and its source/tokenizer manifest;
- `training-metrics.jsonl` heartbeats;
- `checkpoints/step-*` atomic rolling checkpoints.

Each checkpoint contains model weights, AdamW and scheduler state, CPU and CUDA
RNG state, completed optimizer step, samples consumed, token count, and loss.
SIGINT or SIGTERM requests a step-boundary checkpoint. The CLI allows up to two
minutes for that checkpoint before forcibly stopping the process. SIGKILL or a
power loss falls back to the most recent periodic checkpoint.

With AWS execution, checkpoints remain in remote staging during pretraining
and are downloaded when the process ends or cancellation succeeds. The
configured `checkpoint_backup_dir` is mapped into that same remote staging
tree, then copied back to its local destination; it is not a live backup to
another disk or to the laptop. Keep the backup directory outside the run
directory; AWS execution rejects overlapping paths. An interrupted connection
may need manual recovery from the remote path printed in the error before
`--resume` can run.
Set `gpu.maxSeconds` to cover the intended pretrain stage (default 24 hours,
maximum 7 days); reaching the deadline ends that process.

## User service

For unattended operation on a Linux machine running the CLI, run the resumable
command under a user systemd service. Replace every path with an absolute path
and use the installed `tt` path reported by `command -v tt`:

```ini
[Unit]
Description=Tuned Tensor foundation pretraining
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/absolute/project
ExecStart=/absolute/path/to/tt pipeline run --spec /absolute/project/tunedtensor.json --resume /absolute/project/.tuned-tensor/foundation-runs/experiment-1
Restart=on-failure
RestartSec=30
TimeoutStopSec=130
KillSignal=SIGTERM

[Install]
WantedBy=default.target
```

Save it as `~/.config/systemd/user/tt-foundation.service`, then run:

```bash
systemctl --user daemon-reload
systemctl --user enable --now tt-foundation.service
journalctl --user -fu tt-foundation.service
```

Enable user lingering if the job must survive logout. Use a UPS for power-loss
protection. With a local GPU, put `checkpoint_backup_dir` on a different
physical disk or remote mounted filesystem. AWS staging does not provide that
separate-disk guarantee. Before a multi-day run, complete a full-load soak,
stop the service during pretraining, and verify the journal reports a resumed
checkpoint after restart.
