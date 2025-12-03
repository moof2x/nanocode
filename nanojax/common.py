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
        nanojax_dir = Path(os.environ.get("NANOJAX_BASE_DIR"))
    else:
        nanojax_dir = Path.home() / ".cache" / "nanojax"
    nanojax_dir.mkdir(parents=True, exist_ok=True)
    return nanojax_dir


