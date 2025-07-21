## Environments

### Resource Management System

Tools and rewards share a resource pool, which contain a number of environment instances. When acquiring, the pool will return the instance to the tool or reward, and annoate it with the `id`. Later, all requests with the same `id` will obtain the same instance. If requested with a new `id`, the pool will return a new instance or stuck the request if there is no available instance. Until other tools or rewards released the environment, it will go back to the pool. And new requests can be further processed on.