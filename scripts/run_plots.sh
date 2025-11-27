#! /bin/bash
cd /ldap_shared/home/s_ljy/Projects/CVPR2022-Pretrained-ViT-PyTorch # change to project directory
source activate vit_new  # activate conda environment

python draw_plots.py \
  --output_path ./inference_results/model_comparison.png \
  --violin_output_path ./inference_results/violin_distribution.png \
  --csv_files ./inference_results/imagenet_SO32/evaluation_results.csv ./inference_results/exfractal_SO32/evaluation_results.csv ./inference_results/rcdb_SO32/evaluation_results.csv \
  --model_names "ImageNet" "ExFractalDB" "RCDB" \
  --abbrev_file ./dataset/SO32_class_abbrev.csv
