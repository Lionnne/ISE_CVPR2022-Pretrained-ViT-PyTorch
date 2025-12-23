#! /bin/bash

export CUDA_VISIBLE_DEVICES=
export WANDB_API_KEY=
export WANDB_MODE=online

cd /your/path/to/CVPR2022-Pretrained-ViT-PyTorch # change to project directory
source /your/path/to/conda/bin/activate vit_env  # activate conda environment

# Best Model Finetuning Script
# model
MODEL=swin_base_patch4_window7_224.ms_in22k
# name of dataset
DATA_NAME=SO
CLASSES=32
# num of epochs
EPOCHS=40
# initial learning rate
LR=1.0e-4
# path to train dataset
SOURCE_DATASET=./dataset/SO32
# num of GPUs
NGPUS=2
# num of processes per node
NPERNODE=2
# output dir path
OUT_DIR=./check_points/${MODEL}/${CLASSES}/finetune
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=64
# experiment settings
LABEL=adamw_aug
# random seed
SEED=42

# environment variable which is the IP address of the machine in rank 0 (need only for multiple nodes)
# MASTER_ADDR="192.168.1.1"
# or export MASTER_PORT=29500

# mpirun -npernode ${NPERNODE} -np ${NGPUS} \
python finetune.py ${SOURCE_DATASET} \
    --seed ${SEED} \
    --model ${MODEL} --experiment finetune_${MODEL}_${DATA_NAME}${CLASSES}_${LR}_${LABEL}${SEED} \
    --input-size 3 224 224 \
    --epochs ${EPOCHS} --lr ${LR} \
    --smoothing 0 \
    --batch-size ${LOCAL_BS} --opt adamw --weight-decay 1e-4 \
    --sched cosine_iter --warmup-epochs 5 --min-lr 1e-7 \
    --num-classes ${CLASSES} \
    --hflip 0.5 --vflip 0.5 \
    --aa rand-m9-mstd0.5-inc1 \
    --reprob 0.25 --remode pixel --recount 1 \
    --repeated-aug  \
    -j 16 --eval-metric class_avg \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --log-wandb --project-name ISE \
    --pretrained


# Reproduction Finetuning Script
# model
MODEL=deit_base_patch16_224
# name of dataset
DATA_NAME=SO
CLASSES=32
# num of epochs
EPOCHS=40
# initial learning rate
LR=1.0e-3
# path to train dataset
SOURCE_DATASET=./dataset/SO32
# num of GPUs
NGPUS=2
# num of processes per node
NPERNODE=2
# output dir path
OUT_DIR=./check_points/${MODEL}/${CLASSES}/finetune
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=64
# experiment settings
LABEL=reproduction
# random seed
SEED=42
# pretrained model: imagenet, exfractal, rcdb
PRETRAIN=imagenet

# environment variable which is the IP address of the machine in rank 0 (need only for multiple nodes)
# MASTER_ADDR="192.168.1.1"
# or export MASTER_PORT=29500

# mpirun -npernode ${NPERNODE} -np ${NGPUS} \
python finetune.py ${SOURCE_DATASET} \
    --seed ${SEED} \
    --model ${MODEL} --experiment finetune_${PRETRAIN}_${MODEL}_${DATA_NAME}${CLASSES}_${LR}_${LABEL}${SEED} \
    --input-size 3 224 224 \
    --epochs ${EPOCHS} --lr ${LR} \
    --batch-size ${LOCAL_BS} --opt sgd --momentum 0.95 \
    --num-classes ${CLASSES} \
    --hflip 0.5 --vflip 0.5 \
    -j 16 --eval-metric loss \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --log-wandb --project-name ISE \
    --pretrained-path ./ckpts/pretrained_${PRETRAIN}_21k_base.pth.tar \
