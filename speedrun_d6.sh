#!/bin/bash
# bash speedrun_d6.sh
START_TIME=$SECONDS

# all the setup stuff
export OMP_NUM_THREADS=1
export NANOJAX_BASE_DIR="$HOME/.cache/nanojax"
export MODEL_TAG=d6
rm -f /tmp/libtpu_lockfile

export LIBTPU_INIT_ARGS="--xla_tpu_use_bundle_aware_cost_model_for_fusions=false --xla_tpu_scoped_vmem_limit_kib=65536"

# train tokenizer on ~1B characters
python -m data.pretrain -d fineweb-edu -n 5
python -m data.pretrain -d the-stack-v2-dedup -n 2

if [ ! -d "$NANOJAX_BASE_DIR/$MODEL_TAG/tokenizer" ]; then
    python -m scripts.tok_train --max_chars=1000000000 --vocab_size=16384
    python -m scripts.tok_eval
fi

python -u -m scripts.base_train \
    --batch_size=64 \
    --minibatch_size=16 \
    --config=configs.d6 \
    --eval_every=500 \
    --sample_every=500
python -u -m scripts.base_eval --checkpoint=base --minibatch-size=16 --max-per-task=500

# download SFT rollout datasets
ROLLOUTS_DIR="$NANOJAX_BASE_DIR/rollouts"
hf download smohammadi/nanocode-tulu-selfoss-evol --repo-type dataset --local-dir "$ROLLOUTS_DIR/nanocode-tulu-selfoss-evol"
hf download smohammadi/nanocode-long-context --repo-type dataset --local-dir "$ROLLOUTS_DIR/nanocode-long-context"

python -u -m scripts.chat_sft \
    --batch_size=64 \
    --minibatch_size=16 \
    --eval_every=250 \
    --sample_every=250

# download DPO preference datasets
hf download smohammadi/nanocode-tulu-selfoss-evol-preference --repo-type dataset --local-dir "$ROLLOUTS_DIR/nanocode-tulu-selfoss-evol-preference"
hf download smohammadi/nanocode-long-context-preference --repo-type dataset --local-dir "$ROLLOUTS_DIR/nanocode-long-context-preference"

python -u -m scripts.dpo \
    --batch_size=64 \
    --minibatch_size=16 \
    --eval_every=100 \
    --sample_every=100

python -m scripts.report

ELAPSED=$(( SECONDS - START_TIME ))
echo "speedrun_d6 total time: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"
echo "copy reports/d6/ to your local machine, e.g. using scp, then:"
echo "  brew install pandoc"
echo "  pandoc reports/d6/report.md -o reports/d6/report.html --standalone"
echo "to chat with your model:"
echo "  NANOJAX_BASE_DIR=$NANOJAX_BASE_DIR MODEL_TAG=$MODEL_TAG python -m scripts.nanocode --max_tokens=1024"
