# !/bin/bash
SAMPLE_FLAGS="--batch_size 16
--num_samples 16 
--timestep_respacing 250"

MODEL_FLAGS="--attention_resolutions 32,16,8 
--class_cond False 
--diffusion_steps 1000 
--image_size 256 
--learn_sigma True 
--noise_schedule linear 
--num_channels 256 
--num_head_channels 64 
--num_res_blocks 2 
--resblock_updown True 
# --use_fp16 True
--use_scale_shift_norm True"
python3.10 \
    -m debugpy --listen 5678 --wait-for-client \
    -m scrpits.classifier_sample.py \
    $MODEL_FLAGS \
    --classifier_scale 10.0 \
    --classifier_path models/256x256_classifier.pt \
    --model_path models/256x256_diffusion_uncond.pt \
    $SAMPLE_FLAGS
