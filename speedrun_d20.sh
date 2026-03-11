#!/bin/bash
# bash speedrun_d20.sh
START_TIME=$SECONDS

# all the setup stuff
export OMP_NUM_THREADS=1
export NANOCODE_BASE_DIR="$HOME/.cache/nanocode"
export MODEL_TAG=d20
rm -f /tmp/libtpu_lockfile

export LIBTPU_INIT_ARGS="--xla_tpu_use_bundle_aware_cost_model_for_fusions=false --xla_tpu_scoped_vmem_limit_kib=65536"

# train tokenizer on ~2B characters
python -m data.pretrain -d fineweb-edu -n 130
python -m data.pretrain -d the-stack-v2-dedup -n 30

if [ ! -d "$NANOCODE_BASE_DIR/$MODEL_TAG/tokenizer" ]; then
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
    python -m scripts.tok_eval
fi

python -u -m scripts.base_train \
    --batch-size=64 \
    --minibatch-size=2 \
    --config=d20 \
    --eval-every=500 \
    --sample-every=500
python -u -m scripts.base_eval --checkpoint=base --minibatch-size=8

# download SFT rollout datasets
ROLLOUTS_DIR="$NANOCODE_BASE_DIR/rollouts"
hf download smohammadi/nanocode-tulu-selfoss-evol --repo-type dataset --local-dir "$ROLLOUTS_DIR/nanocode-tulu-selfoss-evol"
hf download smohammadi/nanocode-long-context --repo-type dataset --local-dir "$ROLLOUTS_DIR/nanocode-long-context"

python -u -m scripts.agentic_sft \
    --batch-size=64 \
    --minibatch-size=2 \
    --eval-every=500 \
    --sample-every=500

# download DPO preference datasets
hf download smohammadi/nanocode-tulu-selfoss-evol-preference --repo-type dataset --local-dir "$ROLLOUTS_DIR/nanocode-tulu-selfoss-evol-preference"
hf download smohammadi/nanocode-long-context-preference --repo-type dataset --local-dir "$ROLLOUTS_DIR/nanocode-long-context-preference"

python -u -m scripts.dpo \
    --batch-size=32 \
    --minibatch-size=2 \
    --eval-every=100 \
    --sample-every=100

python -m scripts.report

ELAPSED=$(( SECONDS - START_TIME ))
echo "speedrun_d20 total time: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"
echo "copy reports/d20/ to your local machine, e.g. using scp, then:"
echo "  brew install pandoc"
echo "  pandoc reports/d20/report.md -o reports/d20/report.html --standalone"
echo "to chat with your model:"
echo "  NANOCODE_BASE_DIR=$NANOCODE_BASE_DIR MODEL_TAG=$MODEL_TAG python -m scripts.nanocode --max-tokens=2048 --attn-impl=splash"
