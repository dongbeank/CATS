export CUDA_VISIBLE_DEVICES=0

model_name=CATS

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --d_layers 3 \
  --dec_in 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 16 \
  --QAM_end 0.2 \
  --batch_size 128

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_192\
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 192 \
  --d_layers 3 \
  --dec_in 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 16 \
  --QAM_end 0.3 \
  --batch_size 128

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 336 \
  --d_layers 3 \
  --dec_in 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 16 \
  --QAM_end 0.5 \
  --batch_size 128

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 720 \
  --d_layers 3 \
  --dec_in 21 \
  --des 'Exp' \
  --itr 1 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 16 \
  --QAM_end 0.7 \
  --train_epochs 10 \
  --batch_size 128