"""Prefetch the dataset files a behavior spec trains on from the Hugging Face Hub.

Mirrors prefetch.py for models: the files land in the shared Hugging Face cache
(``<HF_HOME>/hub/datasets--org--name``), the revision is pinned to an immutable
commit, and every requested file is checked against its content-addressed blob.
Only the split files named in the spec are downloaded, never the whole repo.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any

from hub_download import configure_download_environment, download_snapshot, verify_blob_digest
from prefetch import configure_hugging_face_cache, write_json

HUGGING_FACE_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")


def validate_repo_path(value: Any) -> str:
    if not isinstance(value, str) or not value or value.startswith(("/", "\\")):
        raise ValueError(f"Dataset file must be a repository-relative path: {value!r}")
    parts = re.split(r"[/\\]", value)
    if any(part in ("", ".", "..") for part in parts) or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"Dataset file must be a repository-relative path: {value!r}")
    return "/".join(parts)


def verify_dataset_files(snapshot: Path, files: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Check each split file is a complete cache entry inside the dataset repo.

    Normal caches link snapshot files to content-addressed blobs, which are
    checked against their digest. Caches without symlink support (Windows
    without Developer Mode) hold plain copies; those are contained and hashed.
    """
    repository = snapshot.parent.parent.resolve()
    verified: dict[str, dict[str, Any]] = {}
    for split, relative in files.items():
        path = snapshot / relative
        if not path.is_file():
            raise ValueError(f"Dataset file {relative} ({split}) is missing from snapshot {snapshot}")
        target = path.resolve()
        if repository not in target.parents:
            raise ValueError(f"Dataset file {relative} escapes the Hugging Face dataset cache: {target}")
        size = target.stat().st_size
        if size == 0:
            raise ValueError(f"Dataset file {relative} ({split}) is empty")
        _, sha256 = verify_blob_digest(path)
        verified[split] = {
            "path": relative,
            "local_path": str(path),
            "size_bytes": size,
            "sha256": sha256,
        }
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch Hugging Face dataset files for a TT behavior spec.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    repo = str(payload["repo"])
    if not HUGGING_FACE_REPO_ID.fullmatch(repo):
        raise ValueError(f"Dataset repo must be a Hugging Face id like org/name: {repo!r}")
    revision = payload.get("revision")
    if revision is not None and not COMMIT_SHA.fullmatch(str(revision)):
        raise ValueError("Dataset revision must be a 40-character Hugging Face commit SHA")
    raw_files = payload.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise ValueError("Dataset prefetch requires at least one split file")
    files = {str(split): validate_repo_path(path) for split, path in raw_files.items()}

    configure_hugging_face_cache(payload.get("model_cache"))
    configure_download_environment()
    from huggingface_hub import constants

    local_only = bool(payload.get("local_files_only", False))
    action = "Verifying cached" if local_only else "Prefetching"
    print(f"{action} dataset {repo} in {constants.HF_HUB_CACHE}...", flush=True)
    snapshot_path, commit = download_snapshot(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=sorted(set(files.values())),
        local_files_only=local_only,
        label="dataset_prefetch",
    )
    snapshot = Path(snapshot_path)
    if not COMMIT_SHA.fullmatch(snapshot.name):
        raise ValueError("Hugging Face dataset snapshot did not resolve to a 40-character immutable commit SHA")
    if revision and snapshot.name.lower() != str(revision).lower():
        raise ValueError(f"Hugging Face returned dataset revision {snapshot.name}, not requested revision {revision}")
    print(f"Verifying dataset files under {snapshot}...", flush=True)
    verified = verify_dataset_files(snapshot, files)

    write_json(args.output, {
        "ok": True,
        "repo": repo,
        "repo_type": "dataset",
        "requested_revision": revision,
        "snapshot_revision": snapshot.name.lower(),
        "snapshot_path": str(snapshot),
        "hf_home": str(constants.HF_HOME),
        "hub_cache": str(constants.HF_HUB_CACHE),
        "files": verified,
        "size_bytes": sum(item["size_bytes"] for item in verified.values()),
        "commit": commit,
    })


if __name__ == "__main__":
    main()
