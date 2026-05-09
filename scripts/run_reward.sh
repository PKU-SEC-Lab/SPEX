#!/bin/bash
set -e
set -x


MODEL_REPO="/opt/models/Llemma-reward-model"

PORT="${PORT:-23456}"
tenser_parellel_size=1

CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model-path $MODEL_REPO --port $PORT --tp-size $tenser_parellel_size --trust-remote-code --mem-fraction-static 0.85

