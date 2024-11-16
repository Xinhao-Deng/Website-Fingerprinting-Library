dataset=CW

for filename in train valid test
  do 
      python -u exp/dataset_process/gen_mtaf.py \
        --dataset ${dataset} \
        --seq_len 10000 \
        --in_file ${filename}
  done

python -u exp/train.py \
  --dataset ${dataset} \
  --model ARES \
  --device cuda:7 \
  --train_file mtaf_train \
  --valid_file mtaf_valid \
  --feature MTAF \
  --seq_len 8000 \
  --train_epochs 300 \
  --batch_size 512 \
  --learning_rate 2e-3 \
  --optimizer AdamW \
  --lradj StepLR \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

python -u exp/test.py \
  --dataset ${dataset} \
  --model ARES \
  --device cuda:7 \
  --valid_file mtaf_valid \
  --test_file mtaf_test  \
  --feature MTAF \
  --seq_len 8000 \
  --batch_size 512 \
  --eval_metrics Accuracy Precision Recall F1-score \
  --load_name max_f1