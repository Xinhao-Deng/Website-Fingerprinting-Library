python -u exp/train.py \
  --dataset DF18 \
  --model AWF \
  --gpu 0 \
  --feature DIR \
  --seq_len 3000 \
  --train_epochs 30 \
  --batch_size 256 \
  --learning_rate 8e-4 \
  --optimizer RMSprop \
  --eval_metrics Accuracy Precision Recall F1-score P@min \
  --save_metric F1-score \
  --save_name max_f1

python -u exp/test.py \
  --dataset DF18 \
  --model AWF \
  --gpu 0 \
  --feature DIR \
  --seq_len 3000 \
  --batch_size 256 \
  --eval_metrics Accuracy Precision Recall F1-score P@min \
  --save_name max_f1