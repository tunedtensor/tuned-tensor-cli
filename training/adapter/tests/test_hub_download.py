"""Shared Hugging Face download behavior: retries, progress and verification."""
import errno
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hub_download  # noqa: E402
from hub_download import (  # noqa: E402
    PROGRESS_PREFIX,
    IncompleteDownloadError,
    PlannedFile,
    ProgressMonitor,
    backoff_seconds,
    emit_progress,
    is_retryable,
    max_attempts_from_env,
    measure,
    plan_files,
    verify_blob,
    with_retries,
    write_ref,
)
from prefetch_dataset import validate_repo_path, verify_dataset_files  # noqa: E402


class RepositoryNotFoundError(Exception):
    pass


class LocalEntryNotFoundError(FileNotFoundError):
    """huggingface_hub raises this when a connection drops on an uncached file."""


class HttpError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


class RetryTests(unittest.TestCase):
    def test_transient_errors_retry_and_caller_errors_fail_fast(self):
        self.assertTrue(is_retryable(ConnectionResetError()))
        self.assertTrue(is_retryable(TimeoutError()))
        self.assertTrue(is_retryable(HttpError(503)))
        self.assertTrue(is_retryable(HttpError(429)))
        self.assertTrue(is_retryable(LocalEntryNotFoundError("connection error")))
        self.assertTrue(is_retryable(IncompleteDownloadError("1 file missing")))
        self.assertFalse(is_retryable(HttpError(401)))
        self.assertFalse(is_retryable(HttpError(404)))
        self.assertFalse(is_retryable(RepositoryNotFoundError()))
        self.assertFalse(is_retryable(ValueError("bad input")))
        self.assertFalse(is_retryable(OSError(errno.ENOSPC, "No space left on device")))

    def test_backoff_is_exponential_and_capped(self):
        self.assertEqual([backoff_seconds(n) for n in (1, 2, 3)], [2, 4, 8])
        self.assertEqual(backoff_seconds(10), 60)

    def test_with_retries_recovers_after_transient_failures(self):
        calls, sleeps, logs = [], [], []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionResetError("wifi dropped")
            return "done"

        result = with_retries(flaky, description="Downloading x", max_attempts=5, sleep=sleeps.append, log=logs.append)
        self.assertEqual(result, "done")
        self.assertEqual(sleeps, [2, 4])
        self.assertIn("attempt 2/5", logs[0])
        self.assertIn("completed files are kept", logs[0])

    def test_with_retries_stops_at_limit_and_on_permanent_errors(self):
        sleeps = []
        with self.assertRaises(ConnectionResetError):
            with_retries(lambda: (_ for _ in ()).throw(ConnectionResetError()), description="x",
                         max_attempts=3, sleep=sleeps.append, log=lambda _: None)
        self.assertEqual(len(sleeps), 2)
        sleeps.clear()
        with self.assertRaises(RepositoryNotFoundError):
            with_retries(lambda: (_ for _ in ()).throw(RepositoryNotFoundError()), description="x",
                         max_attempts=3, sleep=sleeps.append, log=lambda _: None)
        self.assertEqual(sleeps, [])

    def test_max_attempts_env_is_validated(self):
        self.assertEqual(max_attempts_from_env({}), 10)
        self.assertEqual(max_attempts_from_env({"TT_HF_DOWNLOAD_MAX_ATTEMPTS": "3"}), 3)
        for bad in ("0", "101", "many"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                max_attempts_from_env({"TT_HF_DOWNLOAD_MAX_ATTEMPTS": bad})


class ProgressTests(unittest.TestCase):
    def test_plan_uses_lfs_sha256_or_git_blob_id(self):
        siblings = [
            SimpleNamespace(rfilename="model.safetensors", size=10, blob_id="a" * 40, lfs=SimpleNamespace(sha256="b" * 64)),
            SimpleNamespace(rfilename="config.json", size=2, blob_id="c" * 40, lfs=None),
            SimpleNamespace(rfilename="unknown", size=None, blob_id=None, lfs=None),
        ]
        self.assertEqual(plan_files(siblings), [
            PlannedFile("model.safetensors", 10, "b" * 64),
            PlannedFile("config.json", 2, "c" * 40),
        ])

    def test_measure_counts_complete_blobs_and_partial_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            blobs = Path(temporary)
            planned = [PlannedFile("a", 100, "a" * 64), PlannedFile("b", 50, "b" * 40), PlannedFile("c", 10, "c" * 40)]
            (blobs / ("a" * 64)).write_bytes(b"x" * 100)
            (blobs / ("b" * 40 + ".1234abcd.incomplete")).write_bytes(b"x" * 20)
            self.assertEqual(measure(blobs, planned), {
                "completed_bytes": 120, "total_bytes": 160, "files_completed": 1, "files_total": 3,
            })

    def test_monitor_emits_start_and_final_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            blobs = Path(temporary)
            events = []
            planned = [PlannedFile("a", 4, "a" * 40)]
            with ProgressMonitor(blobs, planned, "dataset_prefetch", emit=events.append, interval=60) as monitor:
                (blobs / ("a" * 40)).write_bytes(b"data")
            monitor.snapshot(force=True, phase="downloaded")
            self.assertEqual(events[0]["completed_bytes"], 0)
            self.assertEqual(events[-1], {
                "label": "dataset_prefetch", "phase": "downloaded", "completed_bytes": 4,
                "total_bytes": 4, "files_completed": 1, "files_total": 1,
            })

    def test_progress_line_protocol(self):
        stream = io.StringIO()
        emit_progress({"label": "x", "completed_bytes": 1}, stream)
        line = stream.getvalue()
        self.assertTrue(line.startswith(PROGRESS_PREFIX))
        self.assertEqual(json.loads(line[len(PROGRESS_PREFIX):]), {"label": "x", "completed_bytes": 1})


class CacheTests(unittest.TestCase):
    def test_refs_are_written_atomically_and_contained(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            write_ref(repository, "main", "f" * 40)
            self.assertEqual((repository / "refs" / "main").read_text(), "f" * 40)
            with self.assertRaises(ValueError):
                write_ref(repository, "../../escape", "f" * 40)

    def test_blob_verification_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content = b"hello"
            blob = root / hashlib.sha256(content).hexdigest()
            blob.write_bytes(content)
            self.assertTrue(verify_blob(blob))
            git = root / hashlib.sha1(b"blob 5\0hello").hexdigest()
            git.write_bytes(content)
            self.assertTrue(verify_blob(git))
            plain = root / "plain.txt"
            plain.write_bytes(b"x")
            self.assertFalse(verify_blob(plain))
            blob.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_blob(blob)


class DatasetFileTests(unittest.TestCase):
    def test_repo_paths_must_be_relative_and_contained(self):
        self.assertEqual(validate_repo_path("data/train.jsonl"), "data/train.jsonl")
        for bad in ("", "/etc/passwd", "../x.jsonl", "data/../../x", "C:\\x.jsonl", "data//x", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_repo_path(bad)

    def _snapshot(self, root: Path, files: dict[str, bytes]) -> Path:
        snapshot = root / "datasets--org--name" / "snapshots" / ("a" * 40)
        blobs = root / "datasets--org--name" / "blobs"
        blobs.mkdir(parents=True)
        for name, content in files.items():
            blob = blobs / hashlib.sha256(content).hexdigest()
            blob.write_bytes(content)
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(blob)
        return snapshot

    def test_verified_files_report_sha256_and_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            content = b'{"messages":[]}\n'
            snapshot = self._snapshot(Path(temporary), {"data/train.jsonl": content})
            verified = verify_dataset_files(snapshot, {"training": "data/train.jsonl"})
            self.assertEqual(verified["training"]["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(verified["training"]["size_bytes"], len(content))
            self.assertEqual(verified["training"]["path"], "data/train.jsonl")

    def test_missing_empty_or_escaping_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = self._snapshot(root, {"train.jsonl": b"x"})
            with self.assertRaisesRegex(ValueError, "missing"):
                verify_dataset_files(snapshot, {"validation": "valid.jsonl"})
            outside = root / "outside.jsonl"
            outside.write_bytes(b"x")
            (snapshot / "escape.jsonl").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "escapes"):
                verify_dataset_files(snapshot, {"training": "escape.jsonl"})


class FakeHub:
    """Stand-in for huggingface_hub with a scripted snapshot_download."""

    def __init__(self, hub_cache: Path, commit: str, outcomes: list):
        self.hub_cache = hub_cache
        self.commit = commit
        self.outcomes = outcomes
        self.calls: list[dict] = []
        self.constants = SimpleNamespace(HF_HUB_CACHE=str(hub_cache), HF_HOME=str(hub_cache.parent))

    def install(self, test: unittest.TestCase) -> None:
        hub = SimpleNamespace(constants=self.constants, snapshot_download=self.snapshot_download, HfApi=self.api)
        utils = SimpleNamespace(filter_repo_objects=lambda items, allow_patterns=None, ignore_patterns=None, key=None: list(items))
        previous = {name: sys.modules.get(name) for name in ("huggingface_hub", "huggingface_hub.utils")}
        sys.modules["huggingface_hub"] = hub
        sys.modules["huggingface_hub.utils"] = utils

        def restore():
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        test.addCleanup(restore)

    def api(self, token=None):
        commit = self.commit
        return SimpleNamespace(repo_info=lambda *args, **kwargs: SimpleNamespace(
            sha=commit,
            siblings=[SimpleNamespace(rfilename="train.jsonl", size=4, blob_id="b" * 40, lfs=None)],
        ))

    def snapshot_download(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        snapshot = self.hub_cache / "datasets--org--name" / "snapshots" / self.commit
        snapshot.mkdir(parents=True, exist_ok=True)
        if outcome == "complete":
            (snapshot / "train.jsonl").write_text("data")
        return str(snapshot)


class DownloadSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.hub_cache = Path(self.temporary.name) / "hub"
        self.commit = "c" * 40
        self.sleeps = []
        original_sleep = hub_download.time.sleep
        hub_download.time.sleep = self.sleeps.append
        self.addCleanup(setattr, hub_download.time, "sleep", original_sleep)
        self.output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = self.output
        self.addCleanup(setattr, sys, "stdout", original_stdout)

    def download(self, revision):
        return hub_download.download_snapshot(
            repo_id="org/name", repo_type="dataset", revision=revision,
            allow_patterns=["train.jsonl"], label="dataset_prefetch",
        )

    def test_a_cached_pinned_commit_needs_no_network(self):
        hub = FakeHub(self.hub_cache, self.commit, ["complete"])
        hub.install(self)
        path, commit = self.download(self.commit)
        self.assertEqual(commit, self.commit)
        self.assertTrue(hub.calls[0]["local_files_only"])
        self.assertEqual(len(hub.calls), 1)
        self.assertIn("already cached", self.output.getvalue())

    def test_connection_drops_and_partial_snapshots_are_retried_on_one_commit(self):
        hub = FakeHub(self.hub_cache, self.commit, [
            LocalEntryNotFoundError("connection reset"),
            "partial",
            "complete",
        ])
        hub.install(self)
        path, commit = self.download(None)
        self.assertEqual(commit, self.commit)
        self.assertEqual([call["revision"] for call in hub.calls], [self.commit] * 3)
        self.assertEqual(self.sleeps, [2, 4])
        self.assertIn("IncompleteDownloadError", self.output.getvalue())
        self.assertEqual((self.hub_cache / "datasets--org--name" / "refs" / "main").read_text(), self.commit)
        progress = [json.loads(line[len(PROGRESS_PREFIX):]) for line in self.output.getvalue().splitlines() if line.startswith(PROGRESS_PREFIX)]
        self.assertEqual(progress[-1]["phase"], "downloaded")

    def test_permanent_errors_are_not_retried(self):
        hub = FakeHub(self.hub_cache, self.commit, [RepositoryNotFoundError("no such repo")])
        hub.install(self)
        with self.assertRaises(RepositoryNotFoundError):
            self.download(None)
        self.assertEqual(self.sleeps, [])


if __name__ == "__main__":
    unittest.main()
