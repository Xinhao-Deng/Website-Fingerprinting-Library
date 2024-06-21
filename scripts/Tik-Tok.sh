python -u exp/train.py \
  --dataset DF18 \
  --model TikTok \
  --gpu 2 \
  --feature DT \
  --seq_len 5000 \
  --train_epochs 30 \
  --batch_size 128 \
  --learning_rate 2e-3 \
  --optimizer Adamax \
  --eval_metrics Accuracy Precision Recall F1-score P@min \
  --save_metric F1-score \
  --save_name max_f1

python -u exp/test.py \
  --dataset DF18 \
  --model TikTok \
  --gpu 2 \
  --feature DT \
  --seq_len 5000 \
  --batch_size 256 \
  --eval_metrics Accuracy Precision Recall F1-score P@min \
  --save_name max_f1