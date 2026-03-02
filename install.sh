#!/bin/bash
DEVICE=${1:-tpu}

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate
# install the repo dependencies
uv sync --extra "$DEVICE"
echo "Install completed. If uv was just installed, run:"
echo "  source \$HOME/.local/bin/env"
echo "Then:"
echo "  source .venv/bin/activate && ./speedrun_d24.sh"
