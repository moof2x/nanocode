import hashlib
import os
from pathlib import Path

import jax


def fold_in_str(key: jax.Array, string: str) -> jax.Array:
    # https://github.com/MatX-inc/seqax/blob/main/jax_extra.py#L11
    return jax.random.fold_in(key, int(hashlib.md5(string.encode()).hexdigest()[:8], base=16))
    
def get_base_dir() -> Path:
    # co-locate nanojax intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOJAX_BASE_DIR"):
        nanojax_dir = os.environ.get("NANOJAX_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanojax_dir = os.path.join(cache_dir, "nanojax")
    return Path(nanojax_dir).mkdir(parents=True, exist_ok=True)

def get_data_dir() -> Path:
    # we're going to use identical datasets across our runs, so let's just download them once
    if os.environ.get("NANOJAX_DATA_DIR"):
        data_dir = os.environ.get("NANOJAX_DATA_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        data_dir = os.path.join(cache_dir, "nanojax")
    return Path(data_dir).mkdir(parents=True, exist_ok=True)

