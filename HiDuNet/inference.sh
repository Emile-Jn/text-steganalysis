export DATA_DIR=/data/test_inference/
export OUTPUT_DIR=./outputs/test/

python ./Hierarchical_Bert.py \
  --model_name_or_path bert-base-uncased \
  --do_infer \
  --data_dir $DATA_DIR \
  --log_dir ./LogFiles/ \
  --output_dir $OUTPUT_DIR \
  --num_hidden_layers 2 \
  --num_output_layers 2 \
  --max_seq_length 128 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --learning_rate 5e-5 \
  --weight_decay 0.1 \
  --save_steps 50 \
  --logging_steps 100 \
  --early_stop_steps 20 \
  --warmup_rate 0.06 \
  --evaluate_during_training \
  --overwrite_output_dir