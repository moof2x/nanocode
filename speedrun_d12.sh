#!/bin/bash
# bash speedrun_d12.sh
START_TIME=$SECONDS

# all the setup stuff
export OMP_NUM_THREADS=1
export NANOJAX_BASE_DIR="$HOME/.cache/nanojax"
export MODEL_TAG=d12
rm -f /tmp/libtpu_lockfile

export LIBTPU_INIT_ARGS="--xla_tpu_use_bundle_aware_cost_model_for_fusions=false --xla_tpu_scoped_vmem_limit_kib=65536"

# train tokenizer on ~2B characters
python -m nanojax.dataset -d fineweb-edu -n 40
python -m nanojax.dataset -d the-stack-v2-dedup -n 10

if [ ! -d "$NANOJAX_BASE_DIR/$MODEL_TAG/tokenizer" ]; then
    python -m scripts.tok_train --max_chars=2000000000
    python -m scripts.tok_eval
fi

python -u -m scripts.base_train \
    --batch_size=128 \
    --minibatch_size=64 \
    --config=configs.d12 \
    --accelerator_flops=918e12 \
    --eval_every=500 \
    --sample_every=500
python -u -m scripts.base_eval --checkpoint=base --minibatch-size=8

python -u -m scripts.chat_sft \
    --batch_size=128 \
    --minibatch_size=64 \
    --accelerator_flops=918e12 \
    --eval_every=500 \
    --sample_every=500

python -u -m scripts.dpo \
    --batch_size=32 \
    --minibatch_size=32 \
    --accelerator_flops=918e12 \
    --eval_every=100 \
    --sample_every=100

python -m scripts.report && uvx pandoc reports/d12/report.md -o reports/d12/report.html

ELAPSED=$(( SECONDS - START_TIME ))
echo "speedrun_d12 total time: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"
echo "to view your report, copy reports/d12/ to your local machine, e.g. using scp"
echo "to chat with your model:"
echo "  NANOJAX_BASE_DIR=$NANOJAX_BASE_DIR MODEL_TAG=$MODEL_TAG python -m scripts.nanocode --max_tokens=1024"
