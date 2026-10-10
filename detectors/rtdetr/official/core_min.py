"""Stand-ins for RT-DETR src/core (registry) and src/misc/dist (single process)."""


def register(cls=None, **kwargs):
    if cls is None:
        return lambda c: c
    return cls


def get_world_size():
    return 1


def is_dist_available_and_initialized():
    return False
