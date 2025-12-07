To install:
  - uv sync - installs jax with cuda12 (default)
  - uv sync --extra metal - installs jax-metal for apple silicon
  - uv sync --extra cpu - installs cpu-only jax
  - uv sync --extra tpu - installs jax with tpu support


uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
