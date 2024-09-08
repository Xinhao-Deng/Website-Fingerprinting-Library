dataset=CW

python -u exp/train.py \
  --dataset ${dataset} \
  --model VarCNN \
  --device cuda:3 \
  --feature DT2 \
  --seq_len 5000 \
  --train_epochs 30 \
  --batch_size 50 \
  --learning_rate 1e-3 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

python -u exp/test.py \
  --dataset ${dataset} \
  --model VarCNN \
  --device cuda:3 \
  --feature DT2 \
  --seq_len 5000 \
  --batch_size 256 \
  --eval_metrics Accuracy Precision Recall F1-score \
  --load_name max_f1