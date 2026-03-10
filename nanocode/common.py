"""
Shared utilities for distributed setup, logging, and path resolution.
"""
import os
from pathlib import Path

import jax


def print0(s="", **kwargs):
    if jax.process_index() == 0:
        print(s, **kwargs)


def get_base_dir() -> Path:
    base_dir = Path(os.environ.get("NANOCODE_BASE_DIR", Path.home() / ".cache" / "nanocode"))
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def get_model_dir() -> Path:
    base_dir = get_base_dir()
    tag = os.environ.get("MODEL_TAG")
    if not tag:
        return base_dir
    model_dir = base_dir / tag
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def setup_logging(log_path):
    import sys
    class DiskLogger:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
        def __getattr__(self, name):
            return getattr(self.files[0], name)

    from pathlib import Path
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, 'w')
    sys.stdout = DiskLogger(sys.stdout, f)
    sys.stderr = DiskLogger(sys.stderr, f)


def init_distributed():
    try:
        jax.distributed.initialize()
    except ValueError:
        pass # TODO: fixme
    world_size = jax.device_count()
    mesh = jax.make_mesh((world_size,), ("b",), axis_types=(jax.sharding.AxisType.Explicit,))
    jax.set_mesh(mesh)
    return world_size, mesh
