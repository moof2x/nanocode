"""
Mostly copied from nanochat/scripts/base_eval.py
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile

import jax
import jax.numpy as jnp

from nanojax.common import get_base_dir, print0, init_distributed, setup_logging
from nanojax.tokenizer import get_tokenizer
from nanojax.checkpointing import load_checkpoint, load_model_config
from nanojax.gpt import GPT
from nanojax.core_eval import evaluate_task


eval_bundle_url = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def download_file(url, filename):
    """download a file from url to filename"""
    import urllib.request
    print0(f"downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print0(f"downloaded {filename}")


def place_eval_bundle(file_path):
    base_dir = get_base_dir()
    eval_bundle_dir = base_dir / "eval_bundle"
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_model(model, tokenizer, minibatch_size, compute_dtype, mesh, max_per_task=-1):
    """
    evaluate a base model on the core benchmark.
    """
    base_dir = get_base_dir()
    eval_bundle_dir = base_dir / "eval_bundle"
    if not eval_bundle_dir.exists():
        zip_path = base_dir / "eval_bundle.zip"
        if not zip_path.exists():
            download_file(eval_bundle_url, str(zip_path))
        place_eval_bundle(str(zip_path))

    config_path = eval_bundle_dir / "core.yaml"
    data_base_path = eval_bundle_dir / "eval_data"
    eval_meta_data = eval_bundle_dir / "eval_meta_data.csv"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        data_path = data_base_path / task_meta['dataset_uri']
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, minibatch_size, data, task_meta, compute_dtype, mesh)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='mid', help='Checkpoint to evaluate: base|mid')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per task (-1 = disable)')
    parser.add_argument('--compute-dtype', type=str, default='bfloat16', help='Compute dtype: float32|bfloat16')
    parser.add_argument('--attn-impl', type=str, default="splash", help="Attention backend: splash (TPU) or eager")
    parser.add_argument('--minibatch-size', type=int, default=4, help="Per-device minibatch size")
    args = parser.parse_args()

    world_size, mesh = init_distributed()
    if jax.process_index() == 0:
        compute_dtype = jnp.bfloat16 if args.compute_dtype == 'bfloat16' else jnp.float32

        base_dir = get_base_dir()
        checkpoint_dir = base_dir / f"{args.checkpoint}_checkpoints"
        model_cfg = load_model_config(checkpoint_dir / "model.zarr")

        setup_logging(base_dir / "base_eval" / f"{args.checkpoint}.txt")
        print0(f"Loading model from {checkpoint_dir}")
        rng = jax.random.key(42)
        model = GPT.init(model_cfg, rng, attn_impl=args.attn_impl)
        model = load_checkpoint(checkpoint_dir / "model.zarr", model)

        tokenizer = get_tokenizer()

        out = evaluate_model(model, tokenizer, args.minibatch_size, compute_dtype, mesh, max_per_task=args.max_per_task)
    
        output_csv_path = base_dir / "base_eval" / f"{args.checkpoint}.csv"
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        results = out["results"]
        centered_results = out["centered_results"]
        core_metric = out["core_metric"]

        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")

        print0("="*80)
        print0(f"model: {args.checkpoint}")
        print0("="*80)
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            print0(f.read())

if __name__ == "__main__":
    main()
