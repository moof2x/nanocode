import hashlib
import os
from pathlib import Path

import jax


def print0(s="", **kwargs):
    if jax.process_index() == 0:
        print(s, **kwargs)


def get_base_dir() -> Path:
    # co-locate nanojax intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOJAX_BASE_DIR"):
        nanojax_dir = Path(os.environ.get("NANOJAX_BASE_DIR"))
    else:
        nanojax_dir = Path.home() / ".cache" / "nanojax"
    nanojax_dir.mkdir(parents=True, exist_ok=True)
    return nanojax_dir


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

    f = open(log_path, 'w')
    sys.stdout = DiskLogger(sys.stdout, f)
    sys.stderr = DiskLogger(sys.stderr, f)


def init_distributed():
    world_size = jax.device_count()
    mesh = jax.make_mesh((world_size,), ("b",), axis_types=(jax.sharding.AxisType.Explicit,))
    jax.set_mesh(mesh)
    return world_size, mesh
