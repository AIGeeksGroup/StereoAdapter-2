#!/bin/bash

python train_stereo_da3_codyra_encoder_mono_ss2d_decoder_without_inp_list.py \
  --name stereoadapter2 \
  --train_datasets tartan_air \
  --da3_pretrained depthanything/checkpoints/DA3-BASE \
  --image_size 480 640 \
  --batch_size 8 \
  --num_steps 100000 \
  --train_iters 22 \
  --valid_iters 32 \
  --validation_frequency 10000 \
  --spatial_scale -0.2 0.4 \
  --saturation_range 0 1.4 \
  --n_downsample 2 \
  --lr 0.0001 \
  --use_mde_init \
  --baseline 0.25 \
  --focal 320.0
