
"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

Taken from karparthy/nanochat/nanochat/dataset.py
"""

import argparse
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq
import requests

from nanojax.common import get_base_dir, init_distributed, print0

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames

DATA_DIR = get_base_dir() / "base_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir: Path):
    """ Looks into a data dir and returns full paths to all parquet files. """
    return sorted([data_dir / f for f in data_dir.iterdir() if f.suffix == ".parquet"])

def parquets_iter_batched(dataset: str, split: str, start: int=0, step: int=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert dataset in ["fineweb-edu", "the-stack-v2-dedup"], "dataset must be one of 'fineweb-edu' or 'the-stack-v2-dedup'"
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    data_dir = DATA_DIR / dataset
    parquet_paths = list_parquet_files(data_dir)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index: int, base_url: str, data_dir: Path):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = data_dir / filename
    if filepath.exists():
        print0(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{base_url}/{filename}"
    print0(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath.with_name(filepath.name + ".tmp")
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            temp_path.rename(filepath)
            print0(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print0(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            filepath.unlink(missing_ok=True)
            filepath.with_name(filepath.name + ".tmp").unlink(missing_ok=True)
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print0(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print0(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-d", "--dataset", type=str, choices=["fineweb-edu", "the-stack-v2-dedup"], required=True)
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    if args.dataset == "fineweb-edu":
        base_url = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
        max_shard = 1822
    elif args.dataset == "the-stack-v2-dedup":
        base_url = "https://huggingface.co/datasets/smohammadi/the-stack-v2-python-shuffle/resolve/main"
        max_shard = 79

    data_dir = DATA_DIR / args.dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    num = max_shard + 1 if args.num_files == -1 else min(args.num_files, max_shard + 1)
    ids_to_download = list(range(num))
    init_distributed()
    print0(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print0(f"Target directory: {data_dir}")
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(partial(download_single_file, base_url=base_url, data_dir=data_dir), ids_to_download)

    successful = sum(1 for success in results if success)
    print0(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {data_dir}")
