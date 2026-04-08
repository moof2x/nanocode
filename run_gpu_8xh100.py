"""GPU speedrun for nanocode d12 (135M params) on 8x NVIDIA H100 80GB.

Adapted from run_gpu.py. Key changes for multi-GPU:
- The codebase already supports multi-device via JAX mesh + shard_map (DDP-style).
  init_distributed() creates a mesh over all visible devices automatically.
- Batch sizes are per-device; the dataloader internally multiplies by local_device_count().
  So with 8 GPUs, global_batch = batch_size * 8.
- We set --accelerator-flops to ~990 TFLOPS (H100 BF16) for accurate MFU reporting.
- We increase minibatch-size where memory allows since each H100 has 80GB.
"""
import subprocess
import sys
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NANOCODE_BASE_DIR"] = os.path.expanduser("~/.cache/nanocode")
os.environ["MODEL_TAG"] = "d12"

PYTHON = sys.executable
BASE_DIR = os.environ["NANOCODE_BASE_DIR"]
MODEL_TAG = os.environ["MODEL_TAG"]

# H100 80GB SXM BF16 tensor core peak: ~990 TFLOPS per chip
# (989.4 TFLOPS for dense BF16 matmul on H100 SXM)
H100_FLOPS = "990e12"

def run(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"STEP: {desc}")
    print(f"CMD:  {' '.join(cmd)}")
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0
    print(f"[{desc}] finished in {elapsed:.1f}s (exit code {result.returncode})", flush=True)
    if result.returncode != 0:
        print(f"ERROR: step '{desc}' failed with exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)

start = time.time()

# 1. Download pretraining data
#run([PYTHON, "-m", "data.pretrain", "-d", "fineweb-edu", "-n", "40"],
#    "Download fineweb-edu")
#run([PYTHON, "-m", "data.pretrain", "-d", "the-stack-v2-dedup", "-n", "10"],
#    "Download the-stack-v2-dedup")

# 2. Train tokenizer (if not already done)
tok_dir = os.path.join(BASE_DIR, MODEL_TAG, "tokenizer")
if not os.path.isdir(tok_dir):
    run([PYTHON, "-m", "scripts.tok_train", "--max-chars=2000000000"],
        "Train tokenizer")
    run([PYTHON, "-m", "scripts.tok_eval"],
        "Eval tokenizer")
else:
    print(f"Tokenizer already exists at {tok_dir}, skipping.")

# 3. Pretrain
# batch_size=128 is per-device; global batch = 128 * 8 = 1024.
# minibatch_size=64 means 2 grad accumulation steps per device.
# With 8xH100, this should be very comfortable memory-wise.
run([PYTHON, "-u", "-m", "scripts.base_train",
     "--batch-size=128", "--minibatch-size=64",
     "--config=d12", "--attn-impl=eager",
     "--accelerator-flops", H100_FLOPS,
     "--eval-every=500", "--sample-every=500"],
    "Pretrain d12")

# 4. Base eval
run([PYTHON, "-u", "-m", "scripts.base_eval",
     "--checkpoint=base", "--minibatch-size=8", "--attn-impl=eager"],
    "Base eval")

# 5. Download SFT datasets
#rollouts_dir = os.path.join(BASE_DIR, "rollouts")
#run(["hf", "download", "smohammadi/nanocode-tulu-selfoss-evol",
#     "--repo-type", "dataset", "--local-dir",
#     os.path.join(rollouts_dir, "nanocode-tulu-selfoss-evol")],
#    "Download SFT dataset (tulu-selfoss-evol)")
#run(["hf", "download", "smohammadi/nanocode-long-context",
#     "--repo-type", "dataset", "--local-dir",
#     os.path.join(rollouts_dir, "nanocode-long-context")],
#    "Download SFT dataset (long-context)")

# 6. Agentic SFT
run([PYTHON, "-u", "-m", "scripts.agentic_sft",
     "--batch-size=128", "--minibatch-size=64",
     "--attn-impl=eager",
     "--accelerator-flops", H100_FLOPS,
     "--eval-every=500", "--sample-every=500"],
    "Agentic SFT")

# 7. Download DPO datasets
#run(["hf", "download", "smohammadi/nanocode-tulu-selfoss-evol-preference",
#     "--repo-type", "dataset", "--local-dir",
#     os.path.join(rollouts_dir, "nanocode-tulu-selfoss-evol-preference")],
#    "Download DPO dataset (preference)")
#run(["hf", "download", "smohammadi/nanocode-long-context-preference",
#     "--repo-type", "dataset", "--local-dir",
#     os.path.join(rollouts_dir, "nanocode-long-context-preference")],
#    "Download DPO dataset (long-context preference)")

# 8. DPO
# DPO holds both policy + ref model in memory, so per-device memory is higher.
# batch_size=32, minibatch_size=32 (no grad accum) should be fine with 80GB per GPU.
run([PYTHON, "-u", "-m", "scripts.dpo",
     "--batch-size=32", "--minibatch-size=32",
     "--attn-impl=eager",
     "--accelerator-flops", H100_FLOPS,
     "--eval-every=100", "--sample-every=100"],
    "DPO")

# 9. Report
run([PYTHON, "-m", "scripts.report"],
    "Generate report")

total = time.time() - start
hrs = int(total // 3600)
mins = int((total % 3600) // 60)
secs = int(total % 60)
print(f"\nspeedrun_d12_gpu_8xh100 total time: {hrs}h {mins}m {secs}s")
