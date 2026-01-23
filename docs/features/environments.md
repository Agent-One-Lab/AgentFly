# Environments

We define an `environment` to be the unit resource that we can scale up. An environment plays several roles:

1. The basic unit for isolation: Operations like executing code, writing files will aftect states and possibly cause safety issues. Therefore, AgentFly supports using a docker container as the backend of the environment. Operations can be done inside container for safe and keep different tasks separated.

2. The basic unit for scaling: We can set the pool size in tool or reward definition. This is the maximum number of instances that the resource manager will open and keep for that environment. Scaling can be done by setting a larger pool size.

## Resource Management System

Tools and rewards share a resource pool, which keeps a number of environment instances. When acquiring, the pool will return the instance to the tool or reward, and annoate it with the `id`. Later, all requests with the same `id` will obtain the same instance. If requested with a new `id`, the pool will return a new instance or stuck the request if there is no available instance. Until other tools or rewards released the environment, it will go back to the pool. And new requests can be further processed on.

## Definition

Compared to tools and rewards, environments are more complex since we need environments to be robust enough. Currently you need to define the following interfaces:

- `start`: (asynchronous) Start the environment. (e.g. for docker, start the container.)
- `reset`: (asynchronous) Reset the environment to initial state. This is to avoid the restart while still obtaining a new environment.
- `step`: Main interface to interact with the environment (e.g. for code interpreter, execute the code and return results).
- `aclose` (asynchronous) Close the environment.
