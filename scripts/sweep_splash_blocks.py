#!/usr/bin/env python3
import subprocess
import re
import json
import time
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime

BLOCK_CONFIGS = [
    (128, 128),
    (256, 128),
    (512, 128),
    (1024, 128),
    (256, 256),
    (512, 256),
]

@dataclass
class SweepResult:
    name: str
    block_q: int
    block_compute: int
    mfu_values: List[float]
    mfu_avg: float
    mfu_std: float
    tkps_values: List[float]
    tkps_avg: float
    peak_memory_gb: float
    runtime_sec: float
    status: str
    error_msg: Optional[str] = None

def run_training(
    config_name: str,
    block_q: int,
    block_compute: int,
    config: str = "configs.d24",
    num_steps: int = 5,
    batch_size: int = 32,
    minibatch_size: int = 1,
    accelerator_flops: float = 918e12,
    seq_len: int = 4096,
    timeout_sec: int = 600
) -> tuple:
    env = os.environ.copy()
    env["SPLASH_BLOCK_Q"] = str(block_q)
    env["SPLASH_BLOCK_COMPUTE"] = str(block_compute)

    cmd = [
        sys.executable, "-m", "scripts.base_train",
        f"--num_steps={num_steps}",
        f"--config={config}",
        f"--batch_size={batch_size}",
        f"--minibatch_size={minibatch_size}",
        f"--accelerator_flops={accelerator_flops}",
        "--eval_every=500",
        "--sample_every=500",
    ]

    print(f"testing {config_name} (block_q={block_q}, block_compute={block_compute})...")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=Path(__file__).parent.parent
        )
        runtime = time.time() - start_time
        return result.stdout, result.stderr, result.returncode, runtime
    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        return "", f"timeout after {timeout_sec}s", -1, runtime
    except Exception as e:
        runtime = time.time() - start_time
        return "", str(e), -1, runtime

def parse_mfu(output: str, last_n: int = 3) -> List[float]:
    pattern = r'mfu: ([\d.]+)'
    matches = re.findall(pattern, output)
    mfu_values = [float(m) for m in matches]
    if last_n > 0 and len(mfu_values) > last_n:
        return mfu_values[-last_n:]
    return mfu_values

def parse_tkps(output: str, last_n: int = 3) -> List[float]:
    pattern = r'tkps: (\d+)'
    matches = re.findall(pattern, output)
    tkps_values = [float(m) for m in matches]
    if last_n > 0 and len(tkps_values) > last_n:
        return tkps_values[-last_n:]
    return tkps_values

def parse_peak_memory(output: str) -> float:
    pattern = r'Peak bytes reserved/limit: ([\d.]+)/([\d.]+)'
    matches = re.findall(pattern, output)
    if matches:
        peak_memory = float(matches[-1][0])
        return peak_memory
    return 0.0

def compute_stats(values: List[float]) -> tuple:
    if not values:
        return 0.0, 0.0

    import statistics
    avg = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return avg, std

def run_sweep(
    config: str = "configs.d24",
    num_steps: int = 5,
    batch_size: int = 32,
    minibatch_size: int = 1,
    accelerator_flops: float = 918e12,
    seq_len: int = 4096,
    save_logs: bool = True
) -> List[SweepResult]:
    results = []

    base_dir = os.environ.get("NANOJAX_BASE_DIR", os.path.expanduser("~/.cache/nanojax_splash_sweep"))
    logs_dir = Path(base_dir) / "logs"
    if save_logs:
        logs_dir.mkdir(parents=True, exist_ok=True)

    for block_q, block_compute in BLOCK_CONFIGS:
        # check if valid
        if seq_len % block_q != 0:
            print(f"  skipping block_q={block_q} (doesn't divide seq_len={seq_len})")
            continue
        if block_q % block_compute != 0:
            print(f"  skipping block_q={block_q}, block_compute={block_compute}")
            continue

        config_name = f"bq{block_q}_bc{block_compute}"
        stdout, stderr, returncode, runtime = run_training(
            config_name, block_q, block_compute, config, num_steps,
            batch_size, minibatch_size, accelerator_flops, seq_len
        )

        if save_logs:
            log_file = logs_dir / f"{config_name}.log"
            with open(log_file, "w") as f:
                f.write(f"=== {config_name} ===\n")
                f.write(f"block_q: {block_q}\n")
                f.write(f"block_compute: {block_compute}\n")
                f.write(f"returncode: {returncode}\n")
                f.write(f"runtime: {runtime:.2f}s\n\n")
                f.write("=== STDOUT ===\n")
                f.write(stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(stderr)

        if returncode == 0:
            mfu_values = parse_mfu(stdout, last_n=3)
            tkps_values = parse_tkps(stdout, last_n=3)
            peak_memory = parse_peak_memory(stdout)

            if mfu_values:
                mfu_avg, mfu_std = compute_stats(mfu_values)
                tkps_avg, _ = compute_stats(tkps_values)

                result = SweepResult(
                    name=config_name,
                    block_q=block_q,
                    block_compute=block_compute,
                    mfu_values=mfu_values,
                    mfu_avg=mfu_avg,
                    mfu_std=mfu_std,
                    tkps_values=tkps_values,
                    tkps_avg=tkps_avg,
                    peak_memory_gb=peak_memory,
                    runtime_sec=runtime,
                    status="success"
                )
                print(f"  ✓ mfu_avg={mfu_avg:.2f}% tkps={tkps_avg:.0f} mem={peak_memory:.2f}GB")
            else:
                result = SweepResult(
                    name=config_name,
                    block_q=block_q,
                    block_compute=block_compute,
                    mfu_values=[],
                    mfu_avg=0.0,
                    mfu_std=0.0,
                    tkps_values=[],
                    tkps_avg=0.0,
                    peak_memory_gb=0.0,
                    runtime_sec=runtime,
                    status="failed",
                    error_msg="no mfu values found"
                )
                print(f"  ✗ no mfu values found")
        else:
            error_msg = stderr[-500:] if stderr else "unknown error"
            result = SweepResult(
                name=config_name,
                block_q=block_q,
                block_compute=block_compute,
                mfu_values=[],
                mfu_avg=0.0,
                mfu_std=0.0,
                tkps_values=[],
                tkps_avg=0.0,
                peak_memory_gb=0.0,
                runtime_sec=runtime,
                status="failed",
                error_msg=error_msg
            )
            print(f"  ✗ failed: {error_msg[:100]}")

        results.append(result)

    return results

def save_results(results: List[SweepResult], output_dir: Path, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    successful_results = [r for r in results if r.status == "success"]
    best_result = max(successful_results, key=lambda r: r.mfu_avg) if successful_results else None

    json_data = {
        "metadata": metadata,
        "results": [asdict(r) for r in results],
        "summary": {
            "best_config": best_result.name if best_result else None,
            "best_block_q": best_result.block_q if best_result else None,
            "best_block_compute": best_result.block_compute if best_result else None,
            "best_mfu": best_result.mfu_avg if best_result else 0.0,
        }
    }

    json_path = output_dir / "splash_sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nresults saved to {json_path}")

def main():
    config = os.environ.get("SWEEP_CONFIG", "configs.d24")
    num_steps = int(os.environ.get("SWEEP_NUM_STEPS", "5"))
    batch_size = int(os.environ.get("SWEEP_BATCH_SIZE", "32"))
    minibatch_size = int(os.environ.get("SWEEP_MINIBATCH_SIZE", "1"))
    accelerator_flops = float(os.environ.get("SWEEP_ACCELERATOR_FLOPS", "918e12"))
    seq_len = int(os.environ.get("SWEEP_SEQ_LEN", "4096"))

    base_dir = os.environ.get("NANOJAX_BASE_DIR", os.path.expanduser("~/.cache/nanojax_splash_sweep"))
    output_dir = Path(base_dir) / "results"

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "accelerator_flops": accelerator_flops,
        "num_steps": num_steps,
        "mfu_measurement": "last 3 steps"
    }

    print("="*60)
    print("splash attention block size sweep")
    print("="*60)
    print(f"config: {config}")
    print(f"seq_len: {seq_len}")
    print(f"batch_size: {batch_size}")
    print(f"minibatch_size: {minibatch_size}")
    print(f"accelerator_flops: {accelerator_flops:.2e}")
    print(f"num_steps: {num_steps}")
    print(f"total configs: {len(BLOCK_CONFIGS)}")
    print("="*60)
    print()

    results = run_sweep(config, num_steps, batch_size, minibatch_size, accelerator_flops, seq_len, save_logs=True)
    save_results(results, output_dir, metadata)

    print("\n" + "="*60)
    print("results summary")
    print("="*60)
    print()
    print(f"{'config':<20} {'block_q':<10} {'block_c':<10} {'mfu_avg':<10} {'tkps':<10} {'mem(GB)':<10} {'status':<10}")
    print("-"*90)

    for r in results:
        print(f"{r.name:<20} {r.block_q:<10} {r.block_compute:<10} {r.mfu_avg:<10.2f} {r.tkps_avg:<10.0f} {r.peak_memory_gb:<10.2f} {r.status:<10}")

    successful_results = [r for r in results if r.status == "success"]
    if successful_results:
        best = max(successful_results, key=lambda r: r.mfu_avg)
        print("\nbest configuration:")
        print(f"  {best.name}: block_q={best.block_q}, block_compute={best.block_compute}")
        print(f"  mfu={best.mfu_avg:.2f}%, tkps={best.tkps_avg:.0f}, mem={best.peak_memory_gb:.2f}GB")
        print()
        print("to use this configuration:")
        print(f"  export SPLASH_BLOCK_Q={best.block_q}")
        print(f"  export SPLASH_BLOCK_COMPUTE={best.block_compute}")

    print(f"\nlogs saved to: {base_dir}/logs")

if __name__ == "__main__":
    main()
