import os
import jax
import hashlib

def fold_in_str(key: jax.Array, string: str) -> jax.Array:
    # https://github.com/MatX-inc/seqax/blob/main/jax_extra.py#L11
    return jax.random.fold_in(key, int(hashlib.md5(string.encode()).hexdigest()[:8], base=16))
    
def get_base_dir():
    # co-locate nanojax intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOJAX_BASE_DIR"):
        nanojax_dir = os.environ.get("NANOJAX_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanojax_dir = os.path.join(cache_dir, "nanojax")
    os.makedirs(nanojax_dir, exist_ok=True)
    return nanojax_dir
