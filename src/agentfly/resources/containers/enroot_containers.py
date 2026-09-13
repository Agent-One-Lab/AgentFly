"""Naming for the enroot containers AgentFly creates.

Every enroot container AgentFly starts (local :class:`~agentfly.resources.runner.LocalRunner`
and the Ray-hosted :class:`~.ray_container_resource.RayEnrootContainerActor`) is named
``agentfly-<resource id>`` (or ``agentfly-<random>`` when no id is given). The prefix
makes AgentFly's containers recognizable in ``enroot list`` next to anything else on
the node, and lets a fresh training run sweep the leftovers of a previous one that
died without releasing its resources (a killed job, an OOM, a lost Ray worker):
``agentfly train`` calls :func:`clear_agentfly_containers` before the trainer starts.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

ENROOT_CONTAINER_PREFIX = "agentfly"
ENROOT_CLEANUP_ENV = "AF_ENROOT_CLEANUP_ON_START"


def enroot_container_name(
    resource_id: Optional[str] = None, prefix: str = ENROOT_CONTAINER_PREFIX
) -> str:
    """Return the enroot container name for ``resource_id``: ``<prefix>-<resource_id>``.

    A name that already carries the prefix is returned unchanged, so re-wrapping a
    container name (the resource id of an enroot-backed resource *is* its container
    name) never double-prefixes. Without an id a random ``<prefix>-<hex>`` name is drawn.
    """
    from enroot.utils.utils import random_name

    if not resource_id:
        return random_name(prefix=prefix)
    if resource_id.startswith(prefix + "-"):
        return resource_id
    return f"{prefix}-{resource_id}"


def clear_agentfly_containers() -> List[str]:
    """Force-remove this node's leftover ``agentfly-*`` enroot containers.

    A thin wrapper over :func:`enroot.clear_enroot_containers` scoped to
    :data:`ENROOT_CONTAINER_PREFIX`, meant to run once before training. Opt out
    with ``AF_ENROOT_CLEANUP_ON_START=0`` (e.g. when two training jobs share a node).
    Never raises: a failure (no enroot binary, an old enroot-py without ``prefix``)
    is logged and skipped so it cannot block a training launch. ``enroot list`` is
    per host, so only the node this runs on is swept. Returns the removed names.
    """
    if os.environ.get(ENROOT_CLEANUP_ENV, "1") != "1":
        return []
    try:
        from enroot import clear_enroot_containers

        removed = clear_enroot_containers(prefix=f"{ENROOT_CONTAINER_PREFIX}-")
    except Exception as e:  # noqa: BLE001 — cleanup must never stop a training launch
        logger.warning("enroot cleanup skipped: %s", e)
        return []
    if removed:
        logger.info("removed %d stale enroot container(s): %s", len(removed), ", ".join(removed))
    return removed
