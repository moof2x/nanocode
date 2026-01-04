from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import zarr
from dataclasses import replace
from nanojax.gpt import GPT, GPTConfig
from nanojax.muon import Muon


def save_checkpoint(filename: Path, state: GPT | Muon):
    state_dict, _ = jax.tree.flatten_with_path(state)
    root = zarr.open_group(filename, mode="w")

    if isinstance(state, Muon):
        root.attrs["step"] = int(state.step)
    elif isinstance(state, GPT):
        root.attrs["config"] = state.cfg
    state_dict = [(p, a) for p, a in state_dict if "step" not in jax.tree_util.keystr(p)]
    for path, arr in state_dict:
        root[jax.tree_util.keystr(path)] = np.asarray(arr, dtype=np.float32)

def load_model_config(filename: Path) -> GPTConfig:
    root = zarr.open_group(filename, mode="r")
    return GPTConfig(**root.attrs["config"])

def load_checkpoint(filename: Path, state: GPT | Muon):
    state_dict, treedef = jax.tree.flatten_with_path(state)
    state_dict = [(p, a) for p, a in state_dict if "step" not in jax.tree_util.keystr(p)]
    
    root = zarr.open_group(filename, mode="r")
    new_state_dict = []
    for path, arr in state_dict:
        path = jax.tree_util.keystr(path)
        new_arr = jnp.asarray(root[path], dtype=arr.dtype)
        assert arr.shape == new_arr.shape, f"Expected shape {arr.shape} but got {new_arr.shape} for {path} in {filename}"
        del arr
        new_state_dict.append(new_arr)

    new_state = jax.tree.unflatten(treedef, new_state_dict)
    if isinstance(state, Muon):
        new_state = replace(new_state, step=jnp.array(root.attrs["step"], dtype=jnp.int32))
    return new_state
        
    
