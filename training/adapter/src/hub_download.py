"""Resumable, observable Hugging Face Hub snapshot downloads.

Model and dataset prefetch share this module so both get the same behavior:

* the requested revision is resolved once to an immutable commit, so retries
  can never mix files from two commits when a branch moves mid-download;
* transient network failures are retried with exponential backoff. Completed
  files stay in the Hugging Face cache, so each retry only fetches what is
  still missing;
* progress is emitted as ``@@tt-progress {json}`` lines on stdout, which the
  Node runner renders as a progress bar. huggingface_hub's own tqdm bars are
  disabled because redrawn carriage-return output is unreadable in logs.

huggingface_hub is imported lazily so the pure helpers stay testable with the
system Python.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

PROGRESS_PREFIX = "@@tt-progress "
COMMIT = re.compile(r"[0-9a-fA-F]{40}")
DEFAULT_MAX_ATTEMPTS = 10
MAX_BACKOFF_SECONDS = 60.0
# A per-request read timeout that tolerates slow or briefly stalled Wi-Fi.
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = "60"


@dataclass(frozen=True)
class PlannedFile:
    """One repository file expected in the snapshot, keyed by its cache blob name."""

    path: str
    size: int
    etag: str


def configure_download_environment() -> None:
    """Must run before huggingface_hub is imported; it reads these at import."""
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", DEFAULT_DOWNLOAD_TIMEOUT_SECONDS)


def repo_folder_name(repo_id: str, repo_type: str) -> str:
    return f"{repo_type}s--{repo_id.replace('/', '--')}"


def max_attempts_from_env(environ: dict[str, str] | None = None) -> int:
    raw = (environ if environ is not None else os.environ).get("TT_HF_DOWNLOAD_MAX_ATTEMPTS")
    if raw is None or not raw.strip():
        return DEFAULT_MAX_ATTEMPTS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("TT_HF_DOWNLOAD_MAX_ATTEMPTS must be an integer from 1 to 100") from exc
    if not 1 <= value <= 100:
        raise ValueError("TT_HF_DOWNLOAD_MAX_ATTEMPTS must be an integer from 1 to 100")
    return value


def backoff_seconds(attempt: int) -> float:
    """Delay before retry ``attempt`` (1-based): 2, 4, 8, ... capped at 60s."""
    return min(MAX_BACKOFF_SECONDS, float(2 ** attempt))


class IncompleteDownloadError(RuntimeError):
    """snapshot_download returned a cached folder that still lacks planned files."""


def is_retryable(error: BaseException) -> bool:
    """Retry network and server failures; fail fast on caller or auth errors.

    huggingface_hub reports a connection failure on a file that is not cached
    yet as LocalEntryNotFoundError (a FileNotFoundError), so that and
    IncompleteSnapshotError are retried during downloads.
    """
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return False
    if isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EACCES, errno.EROFS, errno.EDQUOT):
        return False
    name = type(error).__name__
    if name in {
        "RepositoryNotFoundError",
        "GatedRepoError",
        "RevisionNotFoundError",
        "RemoteEntryNotFoundError",
        "DisabledRepoError",
        "OfflineModeIsEnabled",
    }:
        return False
    if isinstance(error, (ValueError, TypeError)):
        return False
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return False
    return True


def plan_files(siblings: Iterable[Any]) -> list[PlannedFile]:
    """Build the expected cache inventory from repo_info(files_metadata=True)."""
    planned: list[PlannedFile] = []
    for sibling in siblings:
        lfs = getattr(sibling, "lfs", None)
        etag = getattr(lfs, "sha256", None) if lfs is not None else None
        etag = etag or getattr(sibling, "blob_id", None)
        size = getattr(sibling, "size", None)
        if not etag or not isinstance(size, int):
            # Without a blob name and size the file cannot be tracked; the
            # download still fetches and verifies it, only progress is coarser.
            continue
        planned.append(PlannedFile(path=sibling.rfilename, size=size, etag=str(etag)))
    return planned


def downloaded_bytes(blobs: Path, planned: PlannedFile) -> tuple[int, bool]:
    """Bytes present for one planned file and whether its blob is complete."""
    if (blobs / planned.etag).is_file():
        return planned.size, True
    partial = 0
    try:
        for candidate in blobs.glob(f"{planned.etag}*.incomplete"):
            try:
                partial = max(partial, candidate.stat().st_size)
            except OSError:
                continue
    except OSError:
        return 0, False
    return min(partial, planned.size), False


def measure(blobs: Path, planned: list[PlannedFile]) -> dict[str, int]:
    completed_bytes = 0
    completed_files = 0
    for item in planned:
        size, done = downloaded_bytes(blobs, item)
        completed_bytes += size
        completed_files += int(done)
    return {
        "completed_bytes": completed_bytes,
        "total_bytes": sum(item.size for item in planned),
        "files_completed": completed_files,
        "files_total": len(planned),
    }


_STDOUT_LOCK = threading.Lock()


def write_line(text: str, stream: Any = None) -> None:
    """Write one whole line; the progress thread and log messages share stdout."""
    target = stream if stream is not None else sys.stdout
    with _STDOUT_LOCK:
        target.write(text + "\n")
        target.flush()


def emit_progress(event: dict[str, Any], stream: Any = None) -> None:
    write_line(PROGRESS_PREFIX + json.dumps(event, separators=(",", ":")), stream)


class ProgressMonitor:
    """Polls the cache blob directory and emits progress while a download runs."""

    def __init__(
        self,
        blobs: Path,
        planned: list[PlannedFile],
        label: str,
        emit: Callable[[dict[str, Any]], None] = emit_progress,
        interval: float = 0.5,
        heartbeat: float = 5.0,
    ) -> None:
        self.blobs = blobs
        self.planned = planned
        self.label = label
        self.emit = emit
        self.interval = interval
        self.heartbeat = heartbeat
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last: dict[str, int] | None = None
        self._last_emit = 0.0

    def snapshot(self, force: bool = False, phase: str = "download") -> None:
        current = measure(self.blobs, self.planned)
        now = time.monotonic()
        if not force and current == self._last and now - self._last_emit < self.heartbeat:
            return
        self._last = current
        self._last_emit = now
        self.emit({"label": self.label, "phase": phase, **current})

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.snapshot()
            except Exception:  # Progress is best effort; never break the download.
                pass

    def __enter__(self) -> "ProgressMonitor":
        self.snapshot(force=True)
        self._thread = threading.Thread(target=self._run, name="tt-progress", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def with_retries(
    action: Callable[[], Any],
    *,
    description: str,
    max_attempts: int,
    sleep: Callable[[float], None] | None = None,
    log: Callable[[str], None] = write_line,
) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            return action()
        except Exception as error:  # noqa: BLE001 - classified below
            if attempt >= max_attempts or not is_retryable(error):
                raise
            delay = backoff_seconds(attempt)
            log(
                f"{description} interrupted ({type(error).__name__}: {error}). "
                f"Retrying in {delay:.0f}s (attempt {attempt + 1}/{max_attempts}); "
                "completed files are kept in the cache."
            )
            (sleep or time.sleep)(delay)
    raise AssertionError("unreachable")


def verify_blob(path: Path) -> bool:
    """Check a cached file against its content-addressed blob name.

    Returns True when the blob name is a recognized digest and it matches;
    False when the name carries no digest (for example a cache without
    symlinks). Raises ValueError on a mismatch.
    """
    return verify_blob_digest(path)[0]


def verify_blob_digest(path: Path) -> tuple[bool, str]:
    """Like verify_blob, also returning the file's SHA-256 from the same read."""
    blob = path.resolve()
    expected = blob.name.lower()
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {blob.stat().st_size}\0".encode())
    with blob.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_sha1.update(chunk)
    digests = {64: sha256.hexdigest(), 40: git_sha1.hexdigest()}
    if len(expected) in digests and all(character in "0123456789abcdef" for character in expected):
        if digests[len(expected)] != expected:
            raise ValueError(f"Cached Hugging Face blob checksum mismatch: {path}")
        return True, digests[64]
    return False, digests[64]


def download_snapshot(
    *,
    repo_id: str,
    repo_type: str,
    revision: str | None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    local_files_only: bool = False,
    label: str,
) -> tuple[str, str]:
    """Download (or verify locally) one snapshot. Returns (path, commit sha)."""
    from huggingface_hub import constants, snapshot_download

    token = os.getenv("HF_TOKEN")
    if local_files_only:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            local_files_only=True,
        )
        return path, Path(path).name

    if revision and COMMIT.fullmatch(revision):
        # A pinned commit that is already fully cached needs no network, so
        # prefetch keeps working offline. huggingface_hub checks completeness
        # against its cached tree listing and raises if files are missing.
        try:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_files_only=True,
            )
            write_line(f"{label}: {repo_id}@{revision} is already cached; no download needed.")
            return path, revision
        except Exception:  # noqa: BLE001 - not cached or incomplete: download below
            pass

    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects

    max_attempts = max_attempts_from_env()
    api = HfApi(token=token)
    info = with_retries(
        lambda: api.repo_info(repo_id, repo_type=repo_type, revision=revision, files_metadata=True),
        description=f"Resolving {repo_id}",
        max_attempts=max_attempts,
    )
    commit = str(info.sha)
    siblings = list(filter_repo_objects(
        info.siblings or [],
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        key=lambda sibling: sibling.rfilename,
    ))
    planned = plan_files(siblings)
    blobs = Path(constants.HF_HUB_CACHE) / repo_folder_name(repo_id, repo_type) / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    total = sum(item.size for item in planned)
    write_line(
        f"{label}: {repo_id}@{commit} — {len(planned)} file(s), {total} bytes. "
        "Interrupted downloads can be restarted; completed files are reused."
    )
    monitor = ProgressMonitor(blobs, planned, label)
    with monitor:
        def attempt() -> str:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=commit,
                token=token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            # If the Hub becomes unreachable mid-way, huggingface_hub may hand
            # back the partial cached folder; treat that as a retryable failure.
            missing = [sibling.rfilename for sibling in siblings if not (Path(path) / sibling.rfilename).exists()]
            if missing:
                raise IncompleteDownloadError(f"{len(missing)} file(s) still missing, e.g. {missing[0]}")
            return path

        path = with_retries(attempt, description=f"Downloading {repo_id}", max_attempts=max_attempts)
    monitor.snapshot(force=True, phase="downloaded")
    if revision and revision != commit:
        # Downloading by commit skips huggingface_hub's ref bookkeeping. Record
        # the branch or tag -> commit mapping it would have written, so later
        # offline resolution of an unpinned revision finds this snapshot.
        write_ref(Path(constants.HF_HUB_CACHE) / repo_folder_name(repo_id, repo_type), revision, commit)
    elif not revision:
        write_ref(Path(constants.HF_HUB_CACHE) / repo_folder_name(repo_id, repo_type), "main", commit)
    return path, commit


def write_ref(repository: Path, revision: str, commit: str) -> None:
    if revision.lower() == commit.lower():
        return
    ref = repository / "refs" / revision
    if repository.resolve() not in ref.resolve().parents:
        raise ValueError(f"Invalid Hugging Face revision name: {revision}")
    ref.parent.mkdir(parents=True, exist_ok=True)
    temporary = ref.with_name(f".{ref.name}.{os.getpid()}.tmp")
    temporary.write_text(commit, encoding="utf-8")
    os.replace(temporary, ref)
