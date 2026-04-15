# Resource Types and Interfaces

## Resource Specification

Declarative configuration used by the engine to create and scale resources:

::: agentfly.resources.types.ResourceSpec
    options:
      show_root_heading: true
      show_members: true
      show_signature: true
      

## Base Resource Interface

Abstract protocol implemented by all managed resource classes:

::: agentfly.resources.types.BaseResource
    options:
      show_root_heading: true
      show_inheritance: true

## Container Resource

Concrete resource wrapper for container-backed runtimes:

::: agentfly.resources.containers.container_resource.ContainerResource
    options:
      show_root_heading: true
      show_inheritance: true

::: agentfly.resources.containers.ray_container_resource.RayContainerResource
    options:
      show_root_heading: true
      show_inheritance: true
