# Resources API Reference

## Overview

AgentFly provides a unified resource system for managing runtime dependencies used by tools and rewards. Environment-like runtimes (for example sandbox containers and simulators) are represented as resources acquired through `Context` and managed by the `ResourceEngine`.

## Components

- [Resource Types and Interfaces](resources.md) - `ResourceSpec`, `BaseResource`, and `ContainerResource`
- [Resource Engine](resource_engine.md) - pooled lifecycle management and backend orchestration
- [Base Environment Resource Class](environment.md) - environment-oriented base interfaces
- [Predefined Environment Resources](predefined_envs.md) - built-in environment resources
