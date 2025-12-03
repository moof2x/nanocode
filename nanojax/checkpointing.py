from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import zarr

from nanojax.gpt import GPT
from nanojax.muon import Muon


def save_checkpoint(filename: Path, state: GPT | Muon):
    state, _ = jax.tree.flatten_with_path(state)
    root = zarr.open_group(filename, mode="w-")
    for path, arr in state:
        root[jax.tree_util.keystr(path)] = np.asarray(arr)


def load_checkpoint(filename: Path, state: GPT | Muon):
    state, treedef = jax.tree.flatten_with_path(state)
    root = zarr.open_group(filename, mode="r")
    new_state = []
    for path, arr in state:
        path = jax.tree_util.keystr(path)
        new_arr = root[path]
        assert arr.shape == new_arr.shape, f"Expected shape {arr.shape} but got {new_arr.shape} for {path} in {filename}"
        assert arr.dtype == new_arr.dtype, f"Expected dtype {arr.dtype} but got {new_arr.dtype} for {path} in {filename}"
        del arr
        new_state.append(jnp.asarray(new_arr))
    return jax.tree.unflatten(treedef, new_state)
        
    
