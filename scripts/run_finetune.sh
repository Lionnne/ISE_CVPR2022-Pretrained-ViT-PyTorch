#! /bin/bash

export CUDA_VISIBLE_DEVICES=3
export WANDB_API_KEY=7599be721549b423244defd25d5bcd0e3b5dc96d
export WANDB_MODE=online
export WANDB_PROJECT=ISE
# export MASTER_PORT=18500 # for multiple expr on the same machine

cd /ldap_shared/home/s_ljy/Projects/CVPR2022-Pretrained-ViT-PyTorch # change to project directory
source activate vit_env  # activate conda environment

CLASSES=32
SAVE_ROOT=./dataset/SO32
LABEL=repro
SEED=666
PRETRAIN=imagenet

# MV-FractalDB Pre-training
# model size
MODEL=base
# initial learning rate
LR=1.0e-3
# name of dataset
DATA_NAME=SO
# num of epochs
EPOCHS=40
# path to train dataset
SOURCE_DATASET=${SAVE_ROOT}
# output dir path
OUT_DIR=./check_points/${MODEL}/${CLASSES}/finetune
# num of GPUs
NGPUS=2
# num of processes per node
NPERNODE=2
# local mini-batch size (global mini-batch size = NGPUS × LOCAL_BS)
LOCAL_BS=64

# environment variable which is the IP address of the machine in rank 0 (need only for multiple nodes)
# MASTER_ADDR="192.168.1.1"

# mpirun -npernode ${NPERNODE} -np ${NGPUS} \
python finetune.py ${SOURCE_DATASET} \
    --seed ${SEED} \
    --model deit_${MODEL}_patch16_224 --experiment finetune_${PRETRAIN}_deit_${MODEL}_${DATA_NAME}${CLASSES}_${LR}_${LABEL}${SEED} \
    --input-size 3 224 224 \
    --epochs ${EPOCHS} --lr ${LR} \
    --smoothing 0 \
    --batch-size ${LOCAL_BS} --opt sgd --momentum 0.95 \
    --num-classes ${CLASSES} \
    --hflip 0.5 --vflip 0.5 \
    -j 16 --eval-metric loss \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --log-wandb \
    --pretrained-path ./ckpts/pretrained_${PRETRAIN}_21k_base.pth.tar \
