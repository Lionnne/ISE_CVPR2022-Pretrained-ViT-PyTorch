#! /bin/bash

cd /ldap_shared/home/s_ljy/Projects/CVPR2022-Pretrained-ViT-PyTorch # change to project directory
conda activate vit_env  # activate conda environment

# number of classes
CLASSES=32
DATA_NAME=SO
LABEL=repro
PRETRAIN=imagenet
SEED=3407
# model size
MODEL=base

# path to test dataset
RESULTS_DIR=./inference_results/${PRETRAIN}_${DATA_NAME}${CLASSES}_${LABEL}${SEED}
CSV_FILE=${RESULTS_DIR}/deit_${MODEL}_patch16_224-224.csv

python evaluation.py \
    --base_path ${RESULTS_DIR} \
    --csv_file ${CSV_FILE} \
    --class_map_file ./dataset/SO32_class_map.txt \
    --abbrev_file ./dataset/SO32_class_abbrev.csv