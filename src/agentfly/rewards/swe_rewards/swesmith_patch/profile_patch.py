"""
Optional patch of swesmith.profiles.base.RepoProfile so pull_image and _cache_image_exists use Enroot.
Also normalizes image_name so profile names like swesmith.architecture.x86_64 match dataset/Docker Hub
(swebench/swesmith.x86_64....) and enroot pulls the correct image.
"""


def _enroot_image_ref(image_name: str) -> str:
    """Convert Docker image name to Enroot ref (e.g. org/repo:tag -> org+repo+tag)."""
    return image_name.replace("/", "+").replace(":", "+")


def _normalize_image_name(name: str) -> str:
    """Use dataset-style image name: swesmith.x86_64 instead of swesmith.architecture.x86_64."""
    if ".architecture.x86_64" in name:
        return name.replace(".architecture.x86_64", ".x86_64", 1)
    if ".architecture.arm64" in name:
        return name.replace(".architecture.arm64", ".arm64", 1)
    return name


def _patch_repo_profile():
    """Patch RepoProfile to use Enroot for image pull and image exists check."""
    import enroot.docker_compat as docker_compat
    from swesmith.profiles.base import RepoProfile

    _enroot_client = None
    _original_image_name = RepoProfile.image_name

    def _get_enroot_client():
        nonlocal _enroot_client
        if _enroot_client is None:
            _enroot_client = docker_compat.from_env()
        return _enroot_client

    @property
    def image_name_normalized(self):
        """Image name normalized to match dataset/Docker Hub (e.g. swesmith.x86_64 not .architecture.x86_64)."""
        return _normalize_image_name(_original_image_name.fget(self))

    def pull_image(self):
        if self._cache_image_exists:
            return
        name = self.image_name  # use normalized name for pull
        try:
            print(f"Pulling image {name}")
            client = _get_enroot_client()
            client.images.pull(name)
        except Exception as e:
            raise RuntimeError(f"Failed to pull image {name}: {e}") from e

    def _cache_image_exists_impl(self):
        try:
            client = _get_enroot_client()
            ref = _enroot_image_ref(self.image_name)
            client.images.get(ref)
            return True
        except Exception:
            return False

    RepoProfile.image_name = image_name_normalized
    RepoProfile.pull_image = pull_image
    # Replace @cached_property _cache_image_exists with a property that uses Enroot
    if "_cache_image_exists" in RepoProfile.__dict__:
        del RepoProfile._cache_image_exists
    RepoProfile._cache_image_exists = property(_cache_image_exists_impl)
