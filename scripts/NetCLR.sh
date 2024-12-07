pretrain_dataset=NCDrift_sup

python -u exp/pretrain.py \
  --dataset ${pretrain_dataset} \
  --model NetCLR \
  --device cuda:6 \
  --train_epochs 100 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --optimizer Adam \
  --save_name pretrain

dataset=NCDrift_inf

python -u exp/train.py \
  --dataset ${dataset} \
  --model NetCLR \
  --device cuda:6 \
  --feature DIR \
  --seq_len 5000 \
  --train_epochs 30 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric Accuracy \
  --load_file checkpoints/${pretrain_dataset}/NetCLR/pretrain.pth \
  --save_name max_f1

python -u exp/test.py \
  --dataset ${dataset} \
  --model NetCLR \
  --device cuda:6 \
  --feature DIR \
  --seq_len 5000 \
  --batch_size 256 \
  --eval_metrics Accuracy Precision Recall F1-score \
  --load_name max_f1
