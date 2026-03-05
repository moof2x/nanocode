To install:
  - uv sync - installs jax with cuda12 (default)
  - uv sync --extra metal - installs jax-metal for apple silicon
  - uv sync --extra cpu - installs cpu-only jax
  - uv sync --extra tpu - installs jax with tpu support
  - for jax nightly on tpu: uv pip install -U --pre jax jaxlib libtpu requests -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ -f https://storage.googleapis.com/jax-releases/libtpu_releases.html


for cuda: TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_MEM_FRACTION=.99
