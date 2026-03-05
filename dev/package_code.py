"""
Repackage Python code from The Stack v2 (smol-ids) into shards:

- each shard is ~100MB in size (after zstd compression)
- parquets are written with row group size of 1024
- two-pass: collect metadata + filter, then shuffle and download content
- content downloaded from Software Heritage's public S3 bucket

This will be uploaded to HuggingFace for hosting.
nanocode's DataLoader will stream and cache shards on disk,
same as the fineweb base_data shards.

NOTE: This file is meant only as reference/documentation of the
dataset preparation and it is not used during the project runtime.
"""
import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config
from datasets import load_dataset
from smart_open import open as smart_open
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--max-chars", type=int, required=True)
parser.add_argument("--min-stars", type=int, default=0)
parser.add_argument("--min-bytes", type=int, default=100)
parser.add_argument("--max-bytes", type=int, default=500_000)
parser.add_argument("--chars-per-shard", type=int, default=250_000_000)
parser.add_argument("--row-group-size", type=int, default=1024)
parser.add_argument("--download-workers", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--push-to-hub", action="store_true", default=False)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
print(f"target: {args.max_chars/1e9:.1f}GB | output: {args.output_dir}")
print(f"filters: min_stars={args.min_stars}, bytes=[{args.min_bytes}, {args.max_bytes}]")

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

def download_content(blob_id, src_encoding):
    try:
        with smart_open(f"s3://softwareheritage/content/{blob_id}", "rb",
                        compression=".gz", transport_params={"client": s3}) as f:
            return f.read().decode(src_encoding or "utf-8", errors="replace")
    except Exception as e:
        print(f"    download failed: {blob_id}: {e}")
        return None

print("\nstreaming metadata, filtering for python...")
ds = load_dataset("bigcode/the-stack-v2-train-smol-ids", split="train", streaming=True)

file_metas = []
total_repos = 0
estimated_chars = 0
t0 = time.time()

for repo in ds:
    total_repos += 1
    stars = repo.get("star_events_count", 0) or 0
    if stars < args.min_stars:
        continue
    for f in repo["files"]:
        if f["language"] != "Python":
            continue
        if f.get("is_vendor") or f.get("is_generated"):
            continue
        length = f.get("length_bytes", 0) or 0
        if length < args.min_bytes or length > args.max_bytes:
            continue
        file_metas.append({"blob_id": f["blob_id"], "src_encoding": f.get("src_encoding") or "utf-8"})
        estimated_chars += length

    if total_repos % 10000 == 0:
        print(f"  {total_repos} repos | {len(file_metas)} python files | ~{estimated_chars/1e9:.2f}GB | {time.time()-t0:.0f}s")
    if estimated_chars >= args.max_chars * 1.2:
        print(f"  collected enough metadata (~1.2x target), stopping early")
        break

nfiles = len(file_metas)
print(f"metadata done: {total_repos} repos, {nfiles} files, ~{estimated_chars/1e9:.2f}GB, {time.time()-t0:.0f}s")

print(f"\nshuffling {nfiles} files, downloading content and writing shards...")
random.Random(args.seed).shuffle(file_metas)

shard_docs = []
shard_index = 0
shard_characters = 0
total_characters = 0
total_files = 0
total_failures = 0
t0 = time.time()

pool = ThreadPoolExecutor(max_workers=args.download_workers)
futures = {pool.submit(download_content, b["blob_id"], b["src_encoding"]): b for b in file_metas}

for future in as_completed(futures):
    content = future.result()
    if content is None:
        total_failures += 1
        continue
    shard_docs.append(content)
    shard_characters += len(content)
    total_characters += len(content)
    total_files += 1

    if total_files % 1000 == 0:
        print(f"  {total_files}/{nfiles} files | {total_characters/1e9:.2f}/{args.max_chars/1e9:.1f}GB | failures: {total_failures}")

    collected_enough = shard_characters >= args.chars_per_shard
    aligned = len(shard_docs) % args.row_group_size == 0
    if collected_enough and aligned:
        shard_path = os.path.join(args.output_dir, f"shard_{shard_index:05d}.parquet")
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(shard_table, shard_path, row_group_size=args.row_group_size,
                       use_dictionary=False, compression="zstd", compression_level=3, write_statistics=False)
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"Wrote {shard_path}. #files: {len(shard_docs)} | #chars: {shard_characters} | "
              f"total: {total_characters/1e9:.2f}/{args.max_chars/1e9:.1f}GB | time: {dt:.2f}s | failures: {total_failures}")
        shard_docs = []
        shard_characters = 0
        shard_index += 1

    if total_characters >= args.max_chars:
        pool.shutdown(wait=False, cancel_futures=True)
        break

if shard_docs:
    shard_path = os.path.join(args.output_dir, f"shard_{shard_index:05d}.parquet")
    shard_table = pa.Table.from_pydict({"text": shard_docs})
    pq.write_table(shard_table, shard_path, row_group_size=args.row_group_size,
                   use_dictionary=False, compression="zstd", compression_level=3, write_statistics=False)
    print(f"Wrote {shard_path}. #files: {len(shard_docs)} | #chars: {shard_characters}")
    shard_index += 1

print(f"\ndone: {total_characters/1e9:.2f}GB, {total_files} files, {shard_index} shards, {total_failures} failures")

if args.push_to_hub:
    from huggingface_hub import HfApi
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    print(f"\nuploading {args.output_dir} to smohammadi/the-stack-v2-python-shuffle...")
    api.upload_large_folder(
        folder_path=args.output_dir,
        repo_id="smohammadi/the-stack-v2-python-shuffle",
        repo_type="dataset",
    )
    print("upload done.")
