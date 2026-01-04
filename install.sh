

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate
# install the repo dependencies
uv sync --extra tpu
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
