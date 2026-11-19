# Datasets and downloads

## Model downloads

`tt models prefetch` shows a progress bar with transfer rate and ETA. Large
downloads survive flaky networks: TT pins the requested revision to one commit,
retries interrupted transfers with backoff (`TT_HF_DOWNLOAD_MAX_ATTEMPTS`,
default 10), and keeps completed files in the Hugging Face cache, so running the
command again resumes instead of starting over.

## Train on a Hugging Face dataset

Point `dataset_prebuilt` at files in a Hugging Face dataset repo instead of
local paths. It works like `base_model`: name the repo, pin a commit, and TT
downloads it into the same Hugging Face cache.

```json
{
  "dataset_prebuilt": {
    "huggingface": {
      "repo": "SetFit/tweet_sentiment_extraction",
      "revision": "0d5359cfbb2470332f2b82e62269cf755e0ac5c3",
      "columns": { "input": "text", "output": "label_text" }
    },
    "training": "train.jsonl",
    "test": "test.jsonl"
  }
}
```

```bash
tt datasets prefetch tunedtensor.json   # download with progress; rerun to resume
tt datasets verify tunedtensor.json     # offline checksum check of the cached files
tt pipeline run --spec tunedtensor.json # downloads automatically if still missing
```

In `tt agent`, ask it to download a dataset or model. It proposes the same
`tt datasets prefetch` or `tt models prefetch` command and runs it after you
approve. The source does not have to be in the spec yet: download first, look at
it, then ask the agent to adopt it, and it proposes a spec edit pinned to the
commit the download resolved. A download alone needs a training filename, not
a validation split or column mapping; those are required when configuring training.
Approval checks the workspace, spec and adjacent legacy runner configuration,
then runs a private snapshot of the reviewed settings. Later edits cannot redirect
the running download. After a failure or cancellation, ask for a new download
proposal to resume from the cache.

## Downloads and the lock file

Any Hugging Face dataset files or model can be downloaded without editing a spec:

```bash
tt datasets prefetch --repo HuggingFaceH4/ultrachat_200k \
  --training data/train_sft-00000-of-00003.parquet --test data/test_sft-00000-of-00001.parquet
tt models prefetch --repo Qwen/Qwen3-4B-Instruct-2507   # add --revision <sha> to pin
```

Direct downloads also work before a spec exists or while it is invalid; those
workspaces use the default cache. An approved agent download reports the commit
from that command's result for a later spec edit.

Files stay in the Hugging Face cache (`HF_HOME`, or `runtime.paths.modelCache`).
Each download also writes a manifest with the repo, commit, file digests and
location under `~/.tuned-tensor/assets/`, so downloads are easy to find later:

```bash
tt assets list                 # everything downloaded, newest first
tt assets path org/name        # where it is on disk (org/name@<sha> for one revision)
tt assets verify [org/name]    # re-hash the files against their manifests
```

`tunedtensor.lock.json` beside the spec pins what a project trains on, the way
`.terraform.lock.hcl` pins providers: the dataset and base-model repo, commit,
and each file's Hugging Face content digest (`sha256:` for LFS files,
`git-sha1:` for small git files). `tt datasets prefetch` and `tt models prefetch`
for the spec write it, and the first real run records any entry it is missing.
After that, `tt pipeline run`:

- uses the locked commit when the spec's dataset does not pin `revision`;
- re-hashes the cached bytes and stops if the spec names another repo or revision,
  a cached file's digest differs, or a locked model file is missing. If the change is intended, rerun the matching
  `prefetch` command for the spec to update the lock.

A first local run with an empty model cache downloads and locks the model before
training starts. Dry runs never download assets or write the lock.

An unpinned dataset stays on its locked commit, including for `tt datasets prefetch`.
To move it to a newer commit, pin that `revision` in the spec and prefetch again.

Commit the lock file with the spec. Runs that load the base model from
`runtime.paths.baseModel` or on an AWS GPU host check only the dataset.

- Split paths are files inside the repo: one `.jsonl` or `.parquet` file per
  split (pick one shard of a sharded split). CSV is not supported yet.
- Parquet supports Snappy, gzip, zstd, Brotli and LZ4 compression, as well as
  uncompressed files.
- Without `columns`, files must already be TT chat JSONL whose system message
  matches the spec. With `columns`, each record's `input` and `output` fields
  become chat rows that use the spec's instruction. Parquet files require
  `columns`. For chat datasets such as `HuggingFaceH4/ultrachat_200k`, use
  `"columns": { "messages": "messages" }`: each conversation's first user turn
  and the assistant reply that follows become one example; later turns are not
  used. Records missing a mapped field are skipped. Evaluation splits keep the first of any duplicate input,
  and training drops inputs that also appear in an evaluation split. The run
  report lists every skipped or dropped count.
- Without `revision`, TT uses the cached `main` snapshot and records the commit
  it resolved. `tt validate` and `tt datasets prefetch` flag the spec as
  unpinned. Pin the commit for reproducible runs.
- Runs never contact the Hub after the download. Each run checks every file
  against its content-addressed blob and records the repo, commit, file
  SHA-256s and conversion counts in `run_metadata.dataset_source`.
- Gated datasets use `HF_TOKEN`, as model downloads do.

## Split a master dataset

Keep converted datasets as unsplit chat JSONL. Downloaded chat JSONL can be
split directly from its verified cache: ask the agent to inspect the download
and use its pinned `split_source`, with no manual copy. Ask it to split a named
file with your ratios and seed, then attach the approved paths to your spec.
For example: “Split data.jsonl 80/20 for training and validation with seed 123,
stratify by metadata.category, and save to split-123.”

```sh
tt datasets inspect data.jsonl
tt datasets split data.jsonl --output split-123 --train .8 --validation .2 --test 0 --seed 123
```

The source stays unchanged. Groups containing repeated prompts stay in training
so held-out prompts remain unique;
optional `--group-by` and `--stratify-by` support family and label boundaries.
See [splitting local data](spec-workflow.md#splitting-an-unsplit-dataset)
for defaults, metadata preservation, and benchmark holdout limitations.
