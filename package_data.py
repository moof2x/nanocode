"""
This is inspired by FlatTokens from seqax.
In this script we're going to be downloading fineweb-edu, tokenizing it
and converting to Zarr format, then writing it to disk in shards, and
finally uploading to the Huggingface Hub for later retrieval.
"""
import os
import time

from datasets import load_dataset

dataset_kwargs = {
    "path": "HuggingFaceFW/fineweb-edu",
    "split": "train",
    "name": "sample-100BT", 
}
ds = load_dataset(**dataset_kwargs)

ds = ds.shuffle(seed=42)
ndocs = len(ds) 
print(f"Total number of documents: {ndocs}")

output_dir = ".cache/nanochat/base_data"
