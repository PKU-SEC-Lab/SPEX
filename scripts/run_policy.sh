#!/bin/bash
set -e
set -x

MODEL_REPO="/opt/models/Llemma-metamath-7b"


PORT="${PORT:-12345}"
tenser_parellel_size=1

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path $MODEL_REPO --port $PORT --tp-size $tenser_parellel_size --trust-remote-code


