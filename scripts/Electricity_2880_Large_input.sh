export CUDA_VISIBLE_DEVICES=0

model_name=CATS

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_2880_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 2880 \
  --pred_len 96 \
  --d_layers 3 \
  --dec_in 321 \
  --des 'Exp' \
  --itr 1 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 16 \
  --QAM_end 0.2 \
  --learning_rate 0.01 \
  --batch_size 32 \
  --use_multi_gpu

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_2880_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 2880 \
  --pred_len 192 \
  --d_layers 3 \
  --dec_in 321 \
  --des 'Exp' \
  --itr 1 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 16 \
  --QAM_end 0.3 \
  --learning_rate 0.01 \
  --batch_size 32 \
  --use_multi_gpu

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_2880_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 2880 \
  --pred_len 336 \
  --d_layers 3 \
  --dec_in 321 \
  --des 'Exp' \
  --itr 1 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 16 \
  --QAM_end 0.5 \
  --learning_rate 0.01 \
  --batch_size 32 \
  --use_multi_gpu

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_2880_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 2880 \
  --pred_len 720 \
  --d_layers 3 \
  --dec_in 321 \
  --des 'Exp' \
  --itr 1 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 16 \
  --QAM_end 0.7 \
  --learning_rate 0.01 \
  --batch_size 32 \
  --use_multi_gpu
