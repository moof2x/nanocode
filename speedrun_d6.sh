#!/bin/bash
# bash speedrun.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# This is also why I hide this script away in dev/

# all the setup stuff
export OMP_NUM_THREADS=1
export NANOJAX_BASE_DIR="$HOME/.cache/nanojax_d6"
mkdir -p $NANOJAX_BASE_DIR

# train tokenizer on ~1B characters
rm -rf "$NANOJAX_BASE_DIR/tokenizer"
python -m nanojax.dataset -n 4
python -m scripts.tok_train --max_chars=10000000000 --vocab_size=32000
python -m scripts.tok_eval
exit 1
# train a very small 4 layer model on the CPU
# each optimization step processes a single sequence of 1024 tokens
# we only run 50 steps of optimization (bump this to get better results)
python -m scripts.base_train \
    --depth=8 \
    --max_seq_len=1024 \
    --device_batch_size=2 \
    --total_batch_size=2048 \
    --eval_every=1000 \
    --eval_tokens=4096 \
    --core_metric_every=1000 \
    --core_metric_max_per_task=12 \
    --sample_every=1000 \
    --num_iterations=1000
python -m scripts.base_loss --device_batch_size=32 --split_tokens=4096
python -m scripts.base_eval --max-per-task=16

# midtraining
python -m scripts.mid_train \
    --max_seq_len=1024 \
    --device_batch_size=16 \
    --eval_every=1000 \
    --eval_tokens=4096 \
    --total_batch_size=16394 \
    --num_iterations=1000
# eval results will be terrible, this is just to execute the code paths.
# note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20

# SFT
python -m scripts.chat_sft \
    --device_batch_size=4 \
    --target_examples_per_step=32 \
    --num_iterations=-1 \
    --eval_steps=100 \
    --num_epochs=1 \
    --eval_metrics_max_problems=16

# Chat CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
# python -m scripts.chat_web

python -m nanochat.report generate
