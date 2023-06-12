#!/bin/bash
TRAIN_FLAGS="--iterations 300
--anneal_lr True
--batch_size 4
--lr 3e-4
--save_interval 100
--weight_decay 0.05"
# CLASSIFIER_FLAGS="--image_size 64
# --classifier_attention_resolutions 32,16,8
# --classifier_depth 2
# --classifier_width 64
# --classifier_pool attention
# --classifier_resblock_updown True
# --classifier_use_scale_shift_norm True"

CLASSIFIER_FLAGS="--image_size 256"

python3.10 \
    -m debugpy --listen 5678 --wait-for-client \
    -m scripts.classifier_train \
    --data_dir datasets/GT-RAIN/GT-RAIN_train \
    --val_data_dir datasets/GT-RAIN/GT-RAIN_val \
    --resume_checkpoint models/resumed/256x256_classifier.pt \
    $TRAIN_FLAGS $CLASSIFIER_FLAGS

#FIXME: dodaj fp16
# FIXME: batchsize:64
