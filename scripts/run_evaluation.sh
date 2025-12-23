#! /bin/bash

cd /your/path/to/CVPR2022-Pretrained-ViT-PyTorch # change to project directory
source /your/path/to/conda/bin/activate vit_env  # activate conda environment

# Best Model Finetuning Script
# model
MODEL=swin_base_patch4_window7_224.ms_in22k
# name of dataset
DATA_NAME=SO
CLASSES=32
# initial learning rate
LR=1.0e-4
# experiment settings
LABEL=adamw_aug
# random seed
SEED=42
# results dir path
RESULTS_DIR=./inference_results/${MODEL}_${DATA_NAME}${CLASSES}_${LR}_${LABEL}${SEED}

python evaluation.py \
    --base_path ${RESULTS_DIR} \
    --csv_file ${RESULTS_DIR}/${MODEL}-224.csv\
    --class_map_file ./dataset/SO32_class_map.txt \
    --abbrev_file ./dataset/SO32_class_abbrev.csv