#!/bin/bash
# quick debugging speedrun on CPU
# bash speedrun_d3_cpu.sh
START_TIME=$SECONDS

# all the setup stuff
source "$HOME/.local/bin/env"
export OMP_NUM_THREADS=1
export NANOJAX_BASE_DIR="$HOME/.cache/nanojax"
export MODEL_TAG=d3

# train tokenizer on ~1B characters
python -m data.pretrain -d fineweb-edu -n 1
python -m data.pretrain -d the-stack-v2-dedup -n 1

if [ ! -d "$NANOJAX_BASE_DIR/$MODEL_TAG/tokenizer" ]; then
    python -m scripts.tok_train --max_chars=1000000000 --vocab_size=8000
    python -m scripts.tok_eval
fi

# python -u -m scripts.base_train \
#     --batch_size=128 \
#     --minibatch_size=128 \
#     --config=configs.d3 \
#     --attn_impl=eager \
#     --num_steps=10 \
#     --eval_every=10 \
#     --sample_every=10
# python -u -m scripts.base_eval --checkpoint=base --minibatch-size=8 --attn-impl=eager

# # download SFT rollout datasets
# ROLLOUTS_DIR="$NANOJAX_BASE_DIR/rollouts"
# hf download smohammadi/nanocode-tulu-selfoss-evol --repo-type dataset --local-dir "$ROLLOUTS_DIR/nanocode-tulu-selfoss-evol"
# hf download smohammadi/nanocode-long-context --repo-type dataset --local-dir "$ROLLOUTS_DIR/nanocode-long-context"

# python -u -m scripts.chat_sft \
#     --batch_size=128 \
#     --minibatch_size=128 \
#     --attn_impl=eager \
#     --num_steps=10 \
#     --eval_every=10 \
#     --sample_every=10

# # download DPO preference datasets
# hf download smohammadi/nanocode-tulu-selfoss-evol-preference --repo-type dataset --local-dir "$ROLLOUTS_DIR/nanocode-tulu-selfoss-evol-preference"
# hf download smohammadi/nanocode-long-context-preference --repo-type dataset --local-dir "$ROLLOUTS_DIR/nanocode-long-context-preference"

# python -u -m scripts.dpo \
#     --batch_size=128 \
#     --minibatch_size=128 \
#     --attn_impl=eager \
#     --num_steps=10 \
#     --eval_every=10 \
#     --sample_every=10

python -m scripts.report

ELAPSED=$(( SECONDS - START_TIME ))
echo "speedrun_d3_cpu total time: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"
echo "copy reports/d3/ to your local machine, e.g. using scp, then:"
echo "  brew install pandoc"
echo "  pandoc reports/d3/report.md -o reports/d3/report.html --standalone"
echo "to chat with your model:"
echo "  NANOJAX_BASE_DIR=$NANOJAX_BASE_DIR MODEL_TAG=$MODEL_TAG python -m scripts.nanocode --max_tokens=1024"
