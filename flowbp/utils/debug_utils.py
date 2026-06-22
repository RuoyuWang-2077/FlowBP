def setup_debug(port=5688):
    import os
    import debugpy
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        debugpy.listen(port)
        debugpy.wait_for_client()
