# nanocode

> The best Claude Code that $200 can buy.

`nanocode` is a library for training your own Claude Code model end-to-end using [Constitutional AI](https://arxiv.org/abs/2212.08073) including tokenizer training, pretraining, synthetic data-generation, agentic SFT with tool use, and DPO with constitutional alignment.
For a detailed writeup, please see the [announcement post](https://github.com/salmanmohammadi/nanocode/discussions/1).

`nanocode` is written in pure JAX and designed to be run on TPUs, and you can get started right away using the [Google TRC program](https://sites.research.google/trc/about/) which gives you free access to TPUs for a month (I think new Google Cloud accounts also get $300 free). You can reproduce nanocode-d24 (1.3B params) in ~9 hours on a TPU v6e-8 for $200, or train nanocode-d20 (~500M params) in ~1.5 hours for $34.

| depth | params | CORE  | cost | time    | MFU   | fwe bpb | sv2 bpb |
|-------|--------|-------|------|---------|-------|---------|---------|
| d12   | 135M   | 0.090 | $3   | 9 min   | 17.4% | 0.956   | 0.689   |
| d20   | 477M   | 0.170 | $34  | 1.4 hrs | 45.2% | 0.838   | 0.533   |
| d24   | 1.3B   | 0.227 | $200 | 9.3 hrs | 52.5% | 0.759   | 0.445   |

### Getting started

If you're following along using TPUs, let's first set up [`gcloud`](https://docs.cloud.google.com/sdk/docs/install-sdk) locally to manage our TPU instances:

```bash
# this is a mac-specific installer - adjust as per your system
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-darwin-arm.tar.gz
tar -xf google-cloud-cli-darwin-arm.tar.gz
./google-cloud-sdk/install.sh
```

You'll then need to do a bit of administration in setting up your Google Cloud project ([see prerequisites #1 and #2 here](https://docs.cloud.google.com/tpu/docs/managing-tpus-tpu-vm#prerequisites)). Once you've done all that, let's spin up a TPU v6e-8 pod:

```bash
export GCLOUD_ID=YOUR_TPU_PROJECT_ID  # from your Google Cloud TPU project setup
gcloud compute tpus tpu-vm create nanocode \
    --zone=us-east1-d \
    --accelerator-type=v6e-8 \
    --version=v2-alpha-tpuv6e \
    --network=default \
    --project=$GCLOUD_ID
    # --spot  uncomment this if you're on the TRC program, as only pre-emptable TPU v6 pods can be provisioned
```

Once your pod is available (check with `gcloud compute tpus tpu-vm list --zone=us-east1-d`), SSH in:

```bash
gcloud compute tpus tpu-vm ssh nanocode --zone=us-east1-d
```

I like to develop on my local machine and use a lightweight `entr` process to synchronise changes through `rsync` to the TPU pod:

```bash
# on your local machine, in a separate tmux/screen/TTY window
git clone git@github.com:salmanmohammadi/nanocode.git
# watches for file changes and automatically pushes to your pod
find ./nanocode \
      \( -path '*/.git' -o -path '*/.venv' -o -path '*/__pycache__' \) -prune \
      -o \( -name '*.py' -o -name '*.toml' -o -name '*.sh' \) -print \
    | entr -r rsync -avz \
      --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
      --exclude='uv.lock' \
      ./nanocode/ -e "gcloud compute tpus tpu-vm ssh nanocode --zone=us-east1-d --" :~/nanocode/
```

But you can also just `git clone` inside your pod directly. Now let's set our environment up:

```bash
# on your TPU pod
cd nanocode/
./install.sh tpu
source .venv/bin/activate
```

Now you can kick off any of the `speedrun` scripts:
```bash
> ls -1 speedrun_
speedrun_d12.sh # 135M params
speedrun_d20.sh # 477M params
speedrun_d24.sh # 1.3B params
speedrun_d3_cpu.sh # 4M params - mostly for debugging on CPU
speedrun_d6.sh # 23M params
```

<details>
<summary><strong>Multi-slice TPU pods</strong></summary>

For larger accelerator configs (e.g. v6e-32), you'll be working with a multi-worker pod, so the steps are a little different:

```bash
gcloud compute tpus tpu-vm create nanocode \
    --zone=europe-west4-a \
    --accelerator-type=v6e-32 \
    --version=v2-alpha-tpuv6e \
    --network=default \
    --project=$GCLOUD_ID \
    --spot
```

Sync your local code to all workers:

```bash
NUM_WORKERS=$(gcloud compute tpus tpu-vm describe nanocode --zone=europe-west4-a --format=json | jq ".networkEndpoints | length")

find ./nanocode \
      \( -path '*/.git' -o -path '*/.venv' -o -path '*/__pycache__' \) -prune \
      -o \( -name '*.py' -o -name '*.toml' -o -name '*.sh' \) -print \
    | entr -r bash -c '
for i in $(seq 0 $((NUM_WORKERS - 1))); do
  rsync -avz --exclude=.git --exclude=.venv --exclude=__pycache__ --exclude=uv.lock \
    ./nanocode/ -e "gcloud compute tpus tpu-vm ssh nanocode --zone=europe-west4-a --worker=$i --" :~/nanocode/ &
done
wait'
```

Install on all workers, then kick off training:

```bash
# install
gcloud compute tpus tpu-vm ssh nanocode --zone=europe-west4-a --worker=all \
    --command="cd ~/nanocode && ./install.sh tpu && source \$HOME/.local/bin/env"

# start training on all workers
gcloud compute tpus tpu-vm ssh nanocode --zone=europe-west4-a --worker=all \
    --command='cd ~/nanocode && tmux new-session -d -s train "source .venv/bin/activate && bash speedrun_d24.sh"'

# monitor worker 0
gcloud compute tpus tpu-vm ssh nanocode --zone=europe-west4-a --worker=0
tmux attach -t train

# stop training on all workers
gcloud compute tpus tpu-vm ssh nanocode --zone=europe-west4-a --worker=all \
    --command='tmux kill-session -t train'
```

You may need to run `ssh-add ~/.ssh/google_compute_engine` if SSH connections to workers fail.

</details>

### Data generation

The synthetic data generation pipeline lives in `dev/`. The restyle and critique passes require either an [OpenRouter](https://openrouter.ai/) API key or a local vLLM server — [this guide](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/inference/trillium/vLLM/Qwen3/README.md) covers spinning up vLLM with Qwen3 on a TPU v6e. If you want to generate your own training data:

- `dev/process_datasets.py` — transforms coding instruction datasets (tulu, selfoss, evol-codealpaca) into tool-call rollouts with soul-guided restyle
- `dev/scenarios_to_rollouts.py` — generates long-context multi-turn agentic rollouts using Gemini with constitutional critique
- `dev/generate_scenarios.py` — generates coding scenarios for rollout generation

Datasets on HuggingFace:
- [smohammadi/nanocode-tulu-selfoss-evol](https://huggingface.co/datasets/smohammadi/nanocode-tulu-selfoss-evol) — ~120K short single-turn rollouts
- [smohammadi/nanocode-long-context](https://huggingface.co/datasets/smohammadi/nanocode-long-context) — ~2K multi-turn agentic rollouts
- [smohammadi/nanocode-tulu-selfoss-evol-preference](https://huggingface.co/datasets/smohammadi/nanocode-tulu-selfoss-evol-preference) — DPO preference pairs
- [smohammadi/nanocode-long-context-preference](https://huggingface.co/datasets/smohammadi/nanocode-long-context-preference) — DPO preference pairs

### Model configs

| config | layers | embed | heads | vocab  | seq len | params |
|--------|--------|-------|-------|--------|---------|--------|
| d3     | 3      | 192   | 2     | 8,000  | 256     | ~4M    |
| d6     | 6      | 384   | 3     | 32,768 | 512     | ~23M   |
| d12    | 12     | 768   | 3     | 32,768 | 1,024   | ~135M  |
| d20    | 20     | 1,280 | 5     | 32,768 | 2,048   | ~477M  |
| d24    | 24     | 2,048 | 8     | 32,768 | 4,096   | ~1.3B  |

### File structure

```
nanocode/
├── nanocode/             # core library
│   ├── gpt.py            # model definition
│   ├── generation.py     # inference engine with KV caching
│   ├── tokenizer.py      # BPE tokenizer
│   ├── configs.py        # model configs (d3-d24)
│   ├── core_eval.py      # CORE evaluation suite
│   ├── dataloader.py     # distributed data loading
│   ├── checkpointing.py  # zarr checkpointing
│   ├── adamw.py          # AdamW optimizer
│   ├── muon.py           # Muon optimizer
│   └── tasks/            # CORE evaluation tasks
├── data/                 # data loading and processing
│   ├── pretrain.py       # fineweb-edu + the-stack-v2 download
│   ├── dataset.py        # base dataset
│   ├── json_dataset.py   # JSON/rollout dataset
│   ├── mixture.py        # dataset mixing
│   └── sequence.py       # sequence packing
├── scripts/              # training and evaluation scripts
│   ├── tok_train.py      # tokenizer training
│   ├── tok_eval.py       # tokenizer evaluation
│   ├── base_train.py     # pretraining
│   ├── base_eval.py      # CORE evaluation
│   ├── agentic_sft.py    # agentic supervised fine-tuning
│   ├── dpo.py            # DPO preference optimization
│   ├── nanocode.py       # CLI agent
│   └── report.py         # training report generation
├── dev/                  # synthetic data generation
│   ├── process_datasets.py
│   ├── scenarios_to_rollouts.py
│   ├── generate_scenarios.py
│   ├── package_code.py
│   └── split_dataset.py
├── speedrun_d*.sh        # end-to-end training scripts
└── install.sh            # environment setup
```

### CPU / GPUs

nanocode has been extensively designed and optimised for TPUs. That said, JAX runs on other backends too:

**CPU**: Use `speedrun_d3_cpu.sh` for debugging the full pipeline locally. It trains a tiny 4M parameter model with `--attn-impl=eager` to bypass TPU-specific splash attention kernels.

**NVIDIA GPUs**: nanocode has been tested on a single NVIDIA GPU but not on multi-GPU setups. You'll need to pass `--attn-impl=eager` to disable splash attention. Install with `./install.sh cuda`.


### Acknowledgements

- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy, which nanocode builds on
- [MaxText](https://github.com/AI-Hypercomputer/maxtext) and [Seqax](https://github.com/MatX-inc/seqax) for excellent references on performant and elegant JAX
- [Google TensorFlow Research Cloud (TRC)](https://sites.research.google/trc/about/) for TPU access
- [HuggingFace](https://huggingface.co/) for [FineWeb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [BigCode](https://www.bigcode-project.org/) for [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) and [self-oss-instruct](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k)
- [AllenAI](https://allenai.org/) for [Tulu-3](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-code)
- [theblackcat102](https://huggingface.co/theblackcat102) for [evol-codealpaca](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1)

### Cite

```bibtex
@misc{nanocode,
  author = {Salman Mohammadi},
  title = {nanocode: The best Claude Code that $200 can buy},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/salmanmohammadi/nanocode}
}
```

### License

MIT
