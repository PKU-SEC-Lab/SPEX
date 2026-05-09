#!/bin/bash
set -e
set -x


POLICY="${POLICY:-12345}"
REWARD="${REWARD:-23456}"

export DATA_DIR="${DATA_DIR:-/data/home/zsz/osdi/rebase/math500.jsonl}"
export OUT_DIR="${OUT_DIR:-./exp_results/}"
export PARA_PATH="${PARA_PATH:-./hype-parameters/bfs.yaml}"
NUM_THREADS="${NUM_THREADS:-1}"
WIDTH="${WIDTH:-8}"

# ---- SPEX defaults: all three optimizations enabled ----
# (1) Intra-query speculation: enabled, with the recommended knobs
SPECULATIVE="${SPECULATIVE:-1}"
export SPEC_LOOKAHEAD="${SPEC_LOOKAHEAD:-3}"
export SPEC_USEFUL_GATE="${SPEC_USEFUL_GATE:-1}"
export SPEC_USEFUL_MIN="${SPEC_USEFUL_MIN:-0.25}"
export SPEC_USEFUL_MIN_FACTOR="${SPEC_USEFUL_MIN_FACTOR:-0.4}"
# (2) Inter-query budget allocation: M = ceil(1.5 * num_threads * width)
export TOTAL_PARALLEL_BUDGET="${TOTAL_PARALLEL_BUDGET:-$(( NUM_THREADS * WIDTH * 3 / 2 ))}"
# (3) Early termination: enabled
EARLY_TERMINATION="${EARLY_TERMINATION:-1}"

EXTRA_FLAGS=()
if [ "$SPECULATIVE" = "1" ]; then EXTRA_FLAGS+=(--speculative); fi
if [ "$EARLY_TERMINATION" = "1" ]; then EXTRA_FLAGS+=(--early_termination); fi
if [ "${ETS:-0}" = "1" ]; then EXTRA_FLAGS+=(--ets); fi
if [ -n "${TRACE_DIR:-}" ]; then EXTRA_FLAGS+=(--trace_dir "$TRACE_DIR"); fi
if [ "${MAX_QUESTIONS:-0}" -gt 0 ]; then EXTRA_FLAGS+=(--max_questions "$MAX_QUESTIONS"); fi

python3 bfs_async.py --input_path $DATA_DIR \
    --output_path $OUT_DIR \
    --parameter_path $PARA_PATH \
    --policy_host http://localhost:$POLICY \
    --reward_host http://localhost:$REWARD \
    --weight_func min \
    --evaluation majority \
    --num_threads "$NUM_THREADS" \
    --width "$WIDTH" \
    "${EXTRA_FLAGS[@]}"
