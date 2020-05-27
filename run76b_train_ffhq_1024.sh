#!/bin/bash
set -ex
export TPU_NAME="${TPU_NAME:-tpu-v3-512-euw4a-53}"
export RUN_NAME="${RUN_NAME:-run76b-ffhq-1024-mirror}"
exec bash run76_train_ffhq_1024.sh

