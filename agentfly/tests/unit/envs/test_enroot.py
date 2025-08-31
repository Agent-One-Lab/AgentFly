from ....envs.manager.enroot import from_env

# Commented out because it's not working on github actions (status is 'exited')
# def test_enroot_client():
#     client = from_env()
#     assert client.ping()
#     container = client.containers.run("nvidia/cuda:11.7.1-devel-ubuntu20.04", "sleep infinity", detach=True)
#     assert container.status == "running"
#     assert container.attrs["State"]["Status"] == "running"
#     assert container.attrs["State"]["Running"] == True

#     container.kill()

