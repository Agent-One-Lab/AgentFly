"""Unit tests for the approval-gated download module (agentfly.utils.download).

These exercise the pure logic (size formatting, env-var gating, path/alias
resolution, approval branching) with the network fully mocked — no HF calls.
"""
import os

import pytest

from agentfly.utils import download


# --------------------------------------------------------------------------- #
# human_size
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "num_bytes, expected",
    [
        (None, "unknown size"),
        (0, "unknown size"),
        (512, "512.0 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (5 * 1024 * 1024, "5.0 MB"),
        (2 * 1024 ** 3, "2.0 GB"),
    ],
)
def test_human_size(num_bytes, expected):
    assert download.human_size(num_bytes) == expected


# --------------------------------------------------------------------------- #
# _assume_yes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", ["1", "true", "yes", "y", "TRUE", " Y "])
def test_assume_yes_true(monkeypatch, value):
    monkeypatch.setenv("AF_ASSUME_YES", value)
    assert download._assume_yes() is True


@pytest.mark.parametrize("value", ["0", "no", "", "off", "false"])
def test_assume_yes_false(monkeypatch, value):
    monkeypatch.setenv("AF_ASSUME_YES", value)
    assert download._assume_yes() is False


def test_assume_yes_unset(monkeypatch):
    monkeypatch.delenv("AF_ASSUME_YES", raising=False)
    assert download._assume_yes() is False


# --------------------------------------------------------------------------- #
# confirm
# --------------------------------------------------------------------------- #


def test_confirm_assume_yes(monkeypatch):
    monkeypatch.setenv("AF_ASSUME_YES", "1")
    assert download.confirm([("file.json", 10)], what="x") is True


def test_confirm_non_interactive_declines(monkeypatch):
    # No AF_ASSUME_YES + non-TTY stdin (pytest) -> declines, never blocks on input().
    monkeypatch.delenv("AF_ASSUME_YES", raising=False)
    assert download.confirm([("file.json", 10)], what="x") is False


# --------------------------------------------------------------------------- #
# _as_list
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        ("a.json", ["a.json"]),
        (["a.json", "b.json"], ["a.json", "b.json"]),
        (["a", 1], ["a", "1"]),
    ],
)
def test_as_list(value, expected):
    assert download._as_list(value) == expected


# --------------------------------------------------------------------------- #
# ensure_training_files
# --------------------------------------------------------------------------- #


def _no_download_sentinel(*args, **kwargs):  # pragma: no cover - must never run
    raise AssertionError("hf_hub_download should not have been called")


def test_ensure_training_files_skips_existing(tmp_path, monkeypatch):
    existing = tmp_path / "already.json"
    existing.write_text("{}")

    # If the file exists, we must not consult the repo or download anything.
    monkeypatch.setattr(download, "_train_repo_files",
                        lambda: (_ for _ in ()).throw(AssertionError("repo consulted")))
    monkeypatch.setattr(download, "hf_hub_download", _no_download_sentinel)

    download.ensure_training_files([str(existing)])
    assert existing.read_text() == "{}"


def test_ensure_training_files_missing_from_repo_is_skipped(tmp_path, monkeypatch):
    # A file that's not in TRAIN_REPO must be reported and left absent, not error.
    monkeypatch.setattr(download, "_train_repo_files", lambda: {})
    monkeypatch.setattr(download, "hf_hub_download", _no_download_sentinel)

    target = tmp_path / "custom_dataset.json"
    download.ensure_training_files([str(target)])  # returns cleanly
    assert not target.exists()


def test_ensure_training_files_declined_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_train_repo_files", lambda: {"foo.json": 123})
    monkeypatch.setattr(download, "confirm", lambda plan, what: False)
    monkeypatch.setattr(download, "hf_hub_download", _no_download_sentinel)

    with pytest.raises(RuntimeError):
        download.ensure_training_files([str(tmp_path / "foo.json")])


def test_ensure_training_files_downloads_when_approved(tmp_path, monkeypatch):
    # Fake HF-cached source the download "returns"; ensure_training_files copies it.
    src = tmp_path / "cached_foo.json"
    src.write_text('{"data": 1}')

    captured = {}

    def fake_download(repo_id, filename, repo_type):
        captured["repo_id"] = repo_id
        captured["filename"] = filename
        return str(src)

    monkeypatch.setattr(download, "_train_repo_files", lambda: {"foo.json": 11})
    monkeypatch.setattr(download, "confirm", lambda plan, what: True)
    monkeypatch.setattr(download, "hf_hub_download", fake_download)

    dest = tmp_path / "nested" / "foo.json"
    download.ensure_training_files([str(dest)])

    assert dest.exists()
    assert dest.read_text() == '{"data": 1}'
    assert captured == {"repo_id": download.TRAIN_REPO, "filename": "foo.json"}


def test_ensure_training_files_resolves_alias(tmp_path, monkeypatch):
    # A local basename that maps to a different repo name via TRAIN_ALIASES.
    src = tmp_path / "cached_remote.json"
    src.write_text("aliased")

    monkeypatch.setattr(download, "TRAIN_ALIASES", {"local.json": "remote.json"})
    monkeypatch.setattr(download, "_train_repo_files", lambda: {"remote.json": 7})
    monkeypatch.setattr(download, "confirm", lambda plan, what: True)
    monkeypatch.setattr(download, "hf_hub_download",
                        lambda repo_id, filename, repo_type: str(src))

    dest = tmp_path / "local.json"
    download.ensure_training_files([str(dest)])
    assert dest.read_text() == "aliased"


# --------------------------------------------------------------------------- #
# ensure_training_data (config wrapper)
# --------------------------------------------------------------------------- #


class _FakeData:
    def __init__(self, mapping):
        self._m = mapping

    def get(self, key, default=None):
        return self._m.get(key, default)


class _FakeConfig:
    def __init__(self, data):
        self.data = data


def test_ensure_training_data_collects_train_and_val(monkeypatch):
    seen = {}
    monkeypatch.setattr(download, "ensure_training_files",
                        lambda paths: seen.setdefault("paths", list(paths)))

    cfg = _FakeConfig(_FakeData({"train_files": "train.json", "val_files": ["val.json"]}))
    download.ensure_training_data(cfg)
    assert seen["paths"] == ["train.json", "val.json"]


def test_ensure_training_data_no_data_section(monkeypatch):
    monkeypatch.setattr(download, "ensure_training_files",
                        lambda paths: (_ for _ in ()).throw(AssertionError("called")))
    download.ensure_training_data(_FakeConfig(None))  # returns cleanly, no crash
