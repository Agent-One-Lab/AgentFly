"""SWE task rewards (R2E-Gym, SWE-bench, SWE-smith).

Importing this package replaces the ``docker`` module in ``sys.modules`` with enroot's Docker
compatibility layer (:func:`patch_docker_with_enroot`), so the SWE evaluation code, written for
the Docker SDK, runs against enroot containers. This applies to the whole process.
"""

import sys


def patch_docker_with_enroot():
    """Patch sys.modules so docker -> enroot.docker_compat, and patch profile for Enroot images."""
    import enroot.docker_compat as docker_compat

    sys.modules["docker"] = docker_compat
    sys.modules["docker.models"] = docker_compat.models
    sys.modules["docker.models.containers"] = docker_compat.models.containers
    sys.modules["docker.errors"] = docker_compat.errors


patch_docker_with_enroot()
