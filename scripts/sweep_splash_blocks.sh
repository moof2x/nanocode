#!/bin/bash
set -e

CONFIG=${1:-"configs.d24"}
NUM_STEPS=${2:-5}
BATCH_SIZE=${3:-32}
MINIBATCH_SIZE=${4:-1}
SEQ_LEN=${5:-4096}
ACCELERATOR_FLOPS=${6:-918e12}

SWEEP_DIR="${NANOJAX_BASE_DIR:-$HOME/.cache/nanojax_splash_sweep}"
LOGS_DIR="$SWEEP_DIR/logs"
mkdir -p "$LOGS_DIR"

BLOCK_CONFIGS=(
    "128:128"
    "256:128"
    "512:128"
    "1024:128"
    "256:256"
    "512:256"
)

echo "========================================"
echo "splash attention block size sweep"
echo "========================================"
echo "config: $CONFIG"
echo "seq_len: $SEQ_LEN"
echo "batch_size: $BATCH_SIZE"
echo "minibatch_size: $MINIBATCH_SIZE"
echo "num_steps: $NUM_STEPS"
echo "accelerator_flops: $ACCELERATOR_FLOPS"
echo "total configs: ${#BLOCK_CONFIGS[@]}"
echo "========================================"
echo ""

RESULTS_FILE="$SWEEP_DIR/results.txt"
> "$RESULTS_FILE"

for config_str in "${BLOCK_CONFIGS[@]}"; do
    IFS=':' read -r block_q block_compute <<< "$config_str"

    # check if valid
    if (( SEQ_LEN % block_q != 0 )); then
        echo "skipping block_q=$block_q (doesn't divide seq_len=$SEQ_LEN)"
        continue
    fi
    if (( block_q % block_compute != 0 )); then
        echo "skipping block_q=$block_q, block_compute=$block_compute (block_q not multiple of block_compute)"
        continue
    fi

    config_name="bq${block_q}_bc${block_compute}"
    echo "testing $config_name (block_q=$block_q, block_compute=$block_compute)..."

    export SPLASH_BLOCK_Q=$block_q
    export SPLASH_BLOCK_COMPUTE=$block_compute

    log_file="$LOGS_DIR/${config_name}.log"

    python -m scripts.base_train \
        --config=$CONFIG \
        --num_steps=$NUM_STEPS \
        --batch_size=$BATCH_SIZE \
        --minibatch_size=$MINIBATCH_SIZE \
        --accelerator_flops=$ACCELERATOR_FLOPS \
        --eval_every=500 \
        --sample_every=500 \
        > "$log_file" 2>&1

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        # parse last 3 mfu values
        mfu_values=$(grep "mfu:" "$log_file" | tail -3 | awk '{print $15}' | tr '\n' ' ')
        tkps_values=$(grep "tkps:" "$log_file" | tail -3 | awk '{print $12}' | tr '\n' ' ')

        if [ -n "$mfu_values" ]; then
            # compute average
            mfu_avg=$(echo "$mfu_values" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
            tkps_avg=$(echo "$tkps_values" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')

            echo "  ✓ mfu_avg=${mfu_avg}% tkps=${tkps_avg}"
            echo "$config_name $block_q $block_compute $mfu_avg $tkps_avg success" >> "$RESULTS_FILE"
        else
            echo "  ✗ no mfu values found"
            echo "$config_name $block_q $block_compute 0 0 failed_no_mfu" >> "$RESULTS_FILE"
        fi
    else
        error_msg=$(tail -5 "$log_file" | head -1)
        echo "  ✗ failed: $error_msg"
        echo "$config_name $block_q $block_compute 0 0 failed_exit_$exit_code" >> "$RESULTS_FILE"
    fi

    unset SPLASH_BLOCK_Q
    unset SPLASH_BLOCK_COMPUTE
    echo ""
done

echo "========================================"
echo "results summary"
echo "========================================"
echo ""
printf "%-20s %-10s %-15s %-10s %-10s %-10s\n" "config" "block_q" "block_compute" "mfu_avg" "tkps" "status"
echo "------------------------------------------------------------------------"

while read -r name bq bc mfu tkps status; do
    printf "%-20s %-10s %-15s %-10s %-10s %-10s\n" "$name" "$bq" "$bc" "$mfu" "$tkps" "$status"
done < "$RESULTS_FILE"

echo ""
echo "best configuration:"
best_line=$(awk '$6 == "success" {print $0}' "$RESULTS_FILE" | sort -k4 -rn | head -1)
if [ -n "$best_line" ]; then
    read -r name bq bc mfu tkps status <<< "$best_line"
    echo "  $name: block_q=$bq, block_compute=$bc, mfu=$mfu%, tkps=$tkps"
    echo ""
    echo "to use this configuration:"
    echo "  export SPLASH_BLOCK_Q=$bq"
    echo "  export SPLASH_BLOCK_COMPUTE=$bc"
else
    echo "  no successful runs"
fi

echo ""
echo "logs saved to: $LOGS_DIR"
echo "results saved to: $RESULTS_FILE"
