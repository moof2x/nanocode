To install:
  - uv sync - installs jax with cuda12 (default)
  - uv sync --extra metal - installs jax-metal for apple silicon
  - uv sync --extra cpu - installs cpu-only jax
  - uv sync --extra tpu - installs jax with tpu support
  - for jax nightly on tpu: uv pip install -U --pre jax jaxlib libtpu requests -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ -f https://storage.googleapis.com/jax-releases/libtpu_releases.html


for cuda: TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_MEM_FRACTION=.99

### File Structure

```
├── data
│   ├── __init__.py
│   ├── common.py
│   ├── dataset.py
│   ├── json_dataset.py
│   ├── mixture.py
│   ├── pretrain.py
│   └── sequence.py
├── dev
│   ├── generate_scenarios.py
│   ├── package_code.py
│   ├── process_datasets.py
│   ├── scenarios_to_rollouts.py
│   └── split_dataset.py
├── nanocode
│   ├── __init__.py
│   ├── tasks
│   ├── adamw.py
│   ├── checkpointing.py
│   ├── common.py
│   ├── configs.py
│   ├── core_eval.py
│   ├── dataloader.py
│   ├── eval.py
│   ├── generation.py
│   ├── gpt.py
│   ├── muon.py
│   └── tokenizer.py
├── scripts
│   ├── agentic_sft.py
│   ├── base_eval.py
│   ├── base_train.py
│   ├── dpo.py
│   ├── nanocode.py
│   ├── report.py
│   ├── tok_eval.py
│   └── tok_train.py
├── LICENSE
├── README.md
├── install.sh
├── motd.txt
├── pyproject.toml
├── speedrun_d12.sh
├── speedrun_d20.sh
├── speedrun_d24.sh
├── speedrun_d3_cpu.sh
├── speedrun_d6.sh
```
