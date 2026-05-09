#!/bin/bash
set -e
set -x


POLICY="${POLICY:-12345}"
REWARD="${REWARD:-23456}"

export DATA_DIR="${DATA_DIR:-/data/home/zsz/osdi/rebase/math500.jsonl}"
export OUT_DIR="${OUT_DIR:-./exp_results}"
export PARA_PATH="${PARA_PATH:-./hype-parameters/bfs.yaml}"
NUM_PATH_LIST="${NUM_PATH_LIST:-5 10 20}"
NUM_THREADS="${NUM_THREADS:-10}"
WID="${WID:-4}"
NUM_SPEC="${NUM_SPEC:-4}"
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"
TRACE_DIR_BASE="${TRACE_DIR_BASE:-}"

# ---- SPEX defaults: all three optimizations enabled ----
# (1) Intra-query speculation + (3) Early termination forwarded as flags.
EXTRA_FLAGS="${EXTRA_FLAGS:---speculative --early_termination}"
# (2) Inter-query budget allocation: M = ceil(0.8 * num_threads * wid * (num_spec+1))
export TOTAL_PARALLEL_BUDGET="${TOTAL_PARALLEL_BUDGET:-$(( NUM_THREADS * WID * (NUM_SPEC + 1) * 4 / 5 ))}"

for NUM_PATH in $NUM_PATH_LIST; do
    EXTRA=""
    if [ -n "$TRACE_DIR_BASE" ]; then
        EXTRA="$EXTRA --trace_dir ${TRACE_DIR_BASE}_np${NUM_PATH}"
    fi
    if [ "$MAX_QUESTIONS" -gt 0 ]; then
        EXTRA="$EXTRA --max_questions $MAX_QUESTIONS"
    fi
    python3 dfs_async.py --input_path $DATA_DIR \
        --output_path $OUT_DIR \
        --parameter_path $PARA_PATH \
        --policy_host http://localhost:$POLICY \
        --reward_host http://localhost:$REWARD \
        --num_path $NUM_PATH  \
        --wid "$WID" \
        --rm orm \
        --num_spec "$NUM_SPEC" \
        --weight_func min \
        --evaluation majority \
        --num_threads "$NUM_THREADS" \
        $EXTRA_FLAGS \
        $EXTRA
done
