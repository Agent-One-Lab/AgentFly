# Resource Engine

## Overview

`ResourceEngine` is the central manager for provisioning, acquiring, releasing, and
ending resources across backends (for example `local`, `ray`, `aws`, and `k8s`).
It maintains pooled free resources and id-bound acquired resources to support reuse
and isolation in multi-rollout execution.

## API Reference

::: agentfly.resources.engine.ResourceEngine
    options:
      show_root_heading: true
      show_members: true
