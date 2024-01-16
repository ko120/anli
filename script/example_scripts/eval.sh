#!/bin/bash

export MASTER_PORT=88888

echo $CUDA_VISIBLE_DEVICES
nvidia-smi
# End visible GPUs.

# setup conda environment
source setup.sh

which python

python src/nli/evaluation.py \
    --model_class_name "bert-base" \
    --max_length 156 \
    --per_gpu_eval_batch_size 16 \
    --model_checkpoint_path saved_models/simcse_anli_128/checkpoints/output/model.pt \
    --eval_data anli_r1_dev:none,anli_r2_dev:none,anli_r3_dev:none \
    --output_prediction_path eval_result
