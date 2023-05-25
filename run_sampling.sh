#!/bin/bash
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 
--class_cond True 
--diffusion_steps 1000 
--dropout 0.1 
--image_size 64 
--learn_sigma True 
--noise_schedule cosine 
--num_channels 192 
--num_head_channels 64 
--num_res_blocks 3 
--resblock_updown True 
--use_new_attention_order True 
--use_fp16 True
--use_scale_shift_norm True"
python3.10 \
    -m debugpy --listen 5678 --wait-for-client \
    -m scripts.classifier_sample \
    $MODEL_FLAGS \
    --classifier_scale 1.0 \
    --classifier_path models/64x64/model000299.pt \
    --classifier_depth 2 \
    --model_path models/64x64_diffusion.pt \
    $SAMPLE_FLAGS
