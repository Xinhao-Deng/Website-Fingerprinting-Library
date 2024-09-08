#!/bin/bash

dataset=CW
attr_method=DeepLiftShap 

for filename in train valid
do 
    python -u exp/data_analysis/temporal_extractor.py \
      --dataset ${dataset} \
      --seq_len 10000 \
      --in_file ${filename}
done

python -u exp/train.py \
  --dataset ${dataset} \
  --model RF \
  --device cuda:6 \
  --train_file temporal_train \
  --valid_file temporal_valid \
  --feature TAM \
  --seq_len 1000 \
  --train_epochs 30 \
  --batch_size 200 \
  --learning_rate 5e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name temporal

python -u exp/data_analysis/feature_attr.py \
  --dataset ${dataset} \
  --model RF \
  --in_file temporal_valid \
  --device cpu \
  --feature TAM \
  --seq_len 1000 \
  --save_name temporal \
  --attr_method ${attr_method}


for filename in train valid
do 
    python -u exp/dataset_process/data_augmentation.py \
      --dataset ${dataset} \
      --model RF \
      --in_file ${filename} \
      --attr_method ${attr_method}
done

for filename in aug_train aug_valid test
do 
    python -u exp/dataset_process/gen_taf.py \
      --dataset ${dataset} \
      --seq_len 10000 \
      --in_file ${filename}
done

python -u exp/train.py \
  --dataset ${dataset} \
  --model Holmes \
  --device cuda:6 \
  --train_file taf_aug_train \
  --valid_file taf_aug_valid \
  --feature TAF \
  --seq_len 2000 \
  --train_epochs 30 \
  --batch_size 256 \
  --learning_rate 5e-4 \
  --loss SupConLoss \
  --optimizer AdamW \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

python -u exp/data_analysis/spatial_analysis.py \
  --dataset ${dataset} \
  --model Holmes \
  --device cuda:6 \
  --valid_file taf_aug_valid \
  --feature TAF \
  --seq_len 2000 \
  --batch_size 256 \
  --save_name max_f1

for percent in {20..100..10}
do
    python -u exp/dataset_process/gen_taf.py \
        --dataset ${dataset} \
        --seq_len 10000 \
        --in_file test_p${percent}

    python -u exp/test.py \
    --dataset ${dataset} \
    --model Holmes \
    --device cuda:6 \
    --valid_file taf_aug_valid \
    --test_file taf_test_p${percent} \
    --feature TAF \
    --seq_len 2000 \
    --batch_size 256 \
    --eval_method Holmes \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    --result_file test_p${percent}
done