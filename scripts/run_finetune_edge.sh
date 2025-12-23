#! /bin/bash

# if you want to try edge-aux finetuning, please run this script

# available model list: 
# swin_aux_{base/small/tiny}_patch4_window7_224
# deit_aux_{base/small/tiny}_patch16_224

# REMENBER: move swin_transformer_aux.py and deit_aux.py to timm/models/ first
# and add the following lines in __init__.py of timm/models/
# from .deit_aux import *
# from .swin_transformer_aux import *

# in args set the --edge-aux flag to enable edge-aux finetuning
# three modes: 
# 1. --edge_aux: use canny edge as an auxiliary branch
# 2. --edge-aux --use-edge-fusion: use canny edge fusion with main branch
# 3. --edge-aux --use-edge-attn: use canny edge as an attention guide (only in deit_aux)
# edge fusion is tested to be the stablest and best way

export CUDA_VISIBLE_DEVICES=
export WANDB_API_KEY=
export WANDB_MODE=online

cd /your/path/to/CVPR2022-Pretrained-ViT-PyTorch # change to project directory
source /your/path/to/conda/bin/activate vit_env  # activate conda environment

# model
MODEL=swin_aux_base_patch4_window7_224
# name of dataset
DATA_NAME=SO
CLASSES=32
# num of epochs
EPOCHS=40
# initial learning rate
LR=1.0e-4
# path to train dataset
SOURCE_DATASET=./dataset/SO32
# output dir path
OUT_DIR=./check_points/${MODEL}/${CLASSES}/finetune
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=64
# experiment settings
LABEL=edge_fusion_aug
# random seed
SEED=42

python finetune_edge.py ${SOURCE_DATASET} \
    --seed ${SEED} \
    --model ${MODEL} --experiment finetune_${MODEL}_${DATA_NAME}${CLASSES}_${LR}_${LABEL}${SEED} \
    --input-size 3 224 224 \
    --epochs ${EPOCHS} --lr ${LR} \
    --smoothing 0 \
    --batch-size ${LOCAL_BS} --opt adamw --weight-decay 1e-4 \
    --sched cosine_iter --warmup-epochs 5 --min-lr 1e-7 \
    --drop-path 0.0 \
    --num-classes ${CLASSES} \
    --hflip 0.5 --vflip 0.5 \
    --aa rand-m9-mstd0.5-inc1 \
    --reprob 0.25 --remode pixel --recount 1 \
    --repeated-aug  \
    --edge-aux --use-edge-fusion \
    -j 16 --eval-metric class_avg \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --log-wandb --project-name ISE \
    --pretrained \
    --no-prefetcher