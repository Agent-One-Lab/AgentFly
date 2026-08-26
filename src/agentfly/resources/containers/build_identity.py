"""Content-addressed build identity for SQS/EKS task images.

Canonical, reusable version of the tag scheme that SkillsScrape's harbor tooling
and the SQS rollout backend both need to agree on, so a prebuilt image is
reused (cache hit) instead of rebuilt. One function, one tag convention.

The tag is a deterministic content hash over the build context (the task's
``environment/`` dir), archived with normalised tarinfo (uid/gid/mtime zeroed,
sorted paths, gzip mtime=0). An optional ``dockerfile_transform`` callable is
applied to the Dockerfile before archiving — the hook image variants use
(e.g. injecting harness tooling); since the transformed Dockerfile is what gets
archived, the variant is captured by the same hash. There is deliberately NO
git-clone-metadata input: no task in the corpus uses ``COPY repo`` /
server-side clone (build-time ``RUN git clone`` is captured via the Dockerfile
text). The historical empty-clone digest suffix is preserved byte-for-byte so
existing ``harbor-v3`` ECR tags stay valid.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Callable, Mapping, Optional

DEFAULT_BUILD_REPOSITORY = "8c8cae2f-3503-468a-be37-5d852496b289"
BUILD_TAG_VERSION = "harbor-v3"

# Preserved from the original scheme: the digest folded in an (always empty)
# clone-metadata JSON. Kept as a constant so tags do not change.
_EMPTY_CLONE_DIGEST_SUFFIX = json.dumps({}, sort_keys=True, separators=(",", ":")).encode()


def _normalise_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    return info


def build_context_archive(
    environment_dir: Path,
    overrides: Optional[Mapping[str, bytes]] = None,
) -> bytes:
    """Deterministic gzip-compressed build context over ``environment_dir``."""
    override_files = dict(overrides or {})
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for path in sorted(
            environment_dir.rglob("*"),
            key=lambda item: item.relative_to(environment_dir).as_posix(),
        ):
            relative = path.relative_to(environment_dir).as_posix()
            info = _normalise_tarinfo(tf.gettarinfo(str(path), arcname=relative))
            if info.isfile():
                if relative in override_files:
                    payload = override_files.pop(relative)
                    info.size = len(payload)
                    tf.addfile(info, io.BytesIO(payload))
                else:
                    with path.open("rb") as source:
                        tf.addfile(info, source)
            else:
                tf.addfile(info)
        if override_files:
            missing = ", ".join(sorted(override_files))
            raise FileNotFoundError(
                f"build-context override targets do not exist: {missing}"
            )

    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0) as gz:
        gz.write(tar_buffer.getvalue())
    return compressed.getvalue()


def build_artifact(
    environment_dir: Path,
    repository: str = "",
    *,
    dockerfile_transform: Optional[Callable[[str], str]] = None,
) -> tuple[str, str, bytes]:
    """Return ``(build_tag, s3_context_key, context_bytes)`` for an image.

    ``dockerfile_transform`` (optional): ``str -> str`` applied to the task's
    Dockerfile before archiving (image-variant hook). ``None`` = no transform.
    """
    repository = repository or DEFAULT_BUILD_REPOSITORY
    if not re.fullmatch(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", repository):
        raise ValueError(f"invalid shared build repository: {repository!r}")
    overrides = None
    if dockerfile_transform is not None:
        dockerfile = (environment_dir / "Dockerfile").read_text()
        overrides = {"Dockerfile": dockerfile_transform(dockerfile).encode()}
    context = build_context_archive(Path(environment_dir), overrides)
    identity = hashlib.sha256()
    identity.update(context)
    identity.update(b"\0")
    identity.update(_EMPTY_CLONE_DIGEST_SUFFIX)
    digest = identity.hexdigest()
    build_tag = f"{repository}:{BUILD_TAG_VERSION}-{digest[:32]}"
    context_key = f"build-contexts/{build_tag}/{digest}.tar.gz"
    return build_tag, context_key, context


def build_identity(
    environment_dir: Path,
    repository: str = "",
    *,
    dockerfile_transform: Optional[Callable[[str], str]] = None,
    registry_url: Optional[str] = None,
) -> dict:
    """Content identity dict (no archive bytes). ``registry_url`` adds
    ``image_uri`` (the pullable ECR reference)."""
    build_tag, context_key, _ = build_artifact(
        environment_dir, repository, dockerfile_transform=dockerfile_transform
    )
    out = {
        "build_tag": build_tag,
        "context_digest": context_key.rsplit("/", 1)[-1].removesuffix(".tar.gz"),
        "build_context_s3_key": context_key,
    }
    if registry_url:
        out["image_uri"] = f"{registry_url.rstrip('/')}/{build_tag}"
    return out
