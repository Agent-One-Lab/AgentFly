"""``agentfly pull-images``: which images it pulls, where it saves them, and how it handles failures."""

import json
import sys
import types

import pytest

from agentfly.utils import pull_images


@pytest.fixture
def fake_enroot(monkeypatch):
    """Replace enroot-py's pull_image with one that writes a small file (or fails for 'bad/*')."""
    pulled = []

    def fake_pull_image(image, registry_host="", directory=None):
        if image.startswith("bad/"):
            raise RuntimeError("[ERROR] 401 Unauthorized")
        pulled.append(f"{registry_host}#{image}" if registry_host else image)
        with open(f"{directory}/{pull_images.image_file_name(image)}", "w") as f:
            f.write("sqsh")

    module = types.ModuleType("enroot.image")
    module.pull_image = fake_pull_image
    monkeypatch.setitem(sys.modules, "enroot.image", module)
    monkeypatch.setattr(pull_images.shutil, "which", lambda _: "/usr/bin/enroot")
    monkeypatch.setattr(pull_images.time, "sleep", lambda _: None)
    return pulled


def write_dataset(tmp_path, images, suffix=".json"):
    path = tmp_path / f"data{suffix}"
    records = [{"image_id": im} for im in images]
    if suffix == ".jsonl":
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    else:
        path.write_text(json.dumps(records))
    return str(path)


def test_file_name_matches_enroot_lookup():
    assert pull_images.image_file_name("namanjain12/aiohttp_final:abc") == "namanjain12+aiohttp_final+abc.sqsh"


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_reads_distinct_images_in_order(tmp_path, suffix):
    path = write_dataset(tmp_path, ["a/x:1", "b/y:2", "a/x:1"], suffix)
    assert pull_images.read_images([path], "image_id") == ["a/x:1", "b/y:2"]


def test_pulls_only_missing_images(tmp_path, fake_enroot):
    out = tmp_path / "images"
    out.mkdir()
    (out / "a+x+1.sqsh").write_text("existing")
    path = write_dataset(tmp_path, ["a/x:1", "b/y:2"])
    assert pull_images.main([path, "--to", str(out)]) == 0
    assert fake_enroot == ["b/y:2"]
    assert sorted(p.name for p in out.iterdir()) == ["a+x+1.sqsh", "b+y+2.sqsh"]  # no .partial left


def test_dry_run_pulls_nothing(tmp_path, fake_enroot):
    path = write_dataset(tmp_path, ["a/x:1"])
    assert pull_images.main([path, "--to", str(tmp_path / "images"), "--dry-run"]) == 0
    assert fake_enroot == []


def test_failures_are_listed_and_cleared_on_success(tmp_path, fake_enroot):
    out = tmp_path / "images"
    path = write_dataset(tmp_path, ["a/x:1", "bad/z:1"])
    assert pull_images.main([path, "--to", str(out), "--retries", "2"]) == 1
    assert (out / "failed_pulls.txt").read_text().split() == ["bad/z:1"]
    assert not (out / "bad+z+1.sqsh").exists()

    # Once every image is present, a rerun succeeds and removes the old failure list.
    (out / "bad+z+1.sqsh").write_text("pulled elsewhere")
    assert pull_images.main([path, "--to", str(out)]) == 0
    assert not (out / "failed_pulls.txt").exists()


def test_registry_is_passed_through_and_names_are_unchanged(tmp_path, fake_enroot):
    out = tmp_path / "images"
    path = write_dataset(tmp_path, ["epoch-research/swe-bench.eval.x86_64.a__b-1"])
    assert pull_images.main([path, "--to", str(out), "--registry", "ghcr.io"]) == 0
    assert fake_enroot == ["ghcr.io#epoch-research/swe-bench.eval.x86_64.a__b-1"]
    assert (out / "epoch-research+swe-bench.eval.x86_64.a__b-1.sqsh").exists()
