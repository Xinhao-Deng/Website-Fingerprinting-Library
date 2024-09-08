dataset=CW

for filename in train valid test
do 
    python -u exp/dataset_process/gen_tam.py \
      --dataset ${dataset} \
      --seq_len 5000 \
      --in_file ${filename}
done

python -u exp/train.py \
  --dataset ${dataset} \
  --model RF \
  --device cuda:0 \
  --train_file tam_train \
  --valid_file tam_valid \
  --feature TAM \
  --seq_len 1800 \
  --train_epochs 30 \
  --batch_size 200 \
  --learning_rate 5e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

python -u exp/test.py \
  --dataset ${dataset} \
  --model RF \
  --device cuda:0 \
  --test_file tam_test \
  --feature TAM \
  --seq_len 1800 \
  --batch_size 256 \
  --eval_metrics Accuracy Precision Recall F1-score \
  --load_name max_f1