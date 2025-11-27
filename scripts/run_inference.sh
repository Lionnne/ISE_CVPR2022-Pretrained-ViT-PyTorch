#! /bin/bash
export CUDA_VISIBLE_DEVICES=6

# number of classes
CLASSES=32
DATA_NAME=SO
LABEL=repro
PRETRAIN=imagenet
SEED=114514
# model size
MODEL=base
# path to test dataset
SOURCE_DATASET=./dataset/SO32/val
# initial learning rate
LR=1.0e-3
# results dir path
RESULTS_DIR=./inference_results/${PRETRAIN}_${DATA_NAME}${CLASSES}_${LABEL}${SEED}

# inference
python inference.py \
    --data-dir ${SOURCE_DATASET} \
    --model deit_${MODEL}_patch16_224 \
    --num-classes ${CLASSES} \
    --checkpoint ./check_points/base/32/finetune/finetune_${PRETRAIN}_deit_${MODEL}_${DATA_NAME}${CLASSES}_${LR}_${LABEL}${SEED}/model_best.pth.tar  \
    --results-dir ${RESULTS_DIR}    \
    --fullname --include-index \
    # --enable-gradcam --viz-dir ./inference_results/${PRETRAIN}_${DATA_NAME}${CLASSES}_${LABEL}${SEED}/vis_results

# evaluation
python evaluation.py \
    --base_path ${RESULTS_DIR} \
    --csv_file ${RESULTS_DIR}/deit_${MODEL}_patch16_224-224.csv\
    --class_map_file ./dataset/SO32_class_map.txt \
    --abbrev_file ./dataset/SO32_class_abbrev.csv