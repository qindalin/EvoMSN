if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi
gpu=0
station_type=adaptive
features=M

for model_name in PatchTST; do
  # for pred_len in 720; do
  for pred_len in 96 192 336 720; do
    python -u run_longExp.py \
      --exp Exp_MSN \
      --is_training 1 \
      --root_path ./datasets/electricity \
      --data_path electricity.csv \
      --model_id electricity_336_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --top_k 4 \
      --batch_size 4 \
      --train_epochs 100\
      --patience 10\
      --station_lr 0.0001 \
      --learning_rate 0.00005 \
      --itr 1 >logs/LongForecasting/$model_name'_electricity_'$pred_len'_'$station_type'_MSN'.log

    python -u run_longExp.py \
      --exp Exp_MSN \
      --station_pretrain_epoch 5 \
      --is_training 1 \
      --root_path ./datasets/exchange_rate \
      --data_path exchange_rate.csv \
      --model_id exchange_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --top_k 4 \
      --itr 1 >logs/LongForecasting/$model_name'_exchange_rate_'$pred_len'_'$station_type'_MSN'.log

    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/traffic \
      --data_path traffic.csv \
      --model_id traffic_336_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --itr 1 \
      --gpu $gpu \
      --period_len 24 \
      --top_k 4 \
      --batch_size 4 \
      --train_epochs 100\
      --patience 10\
      --station_type $station_type >logs/LongForecasting/$model_name'_traffic_'$pred_len'_'$station_type'_MSN'.log

    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/weather \
      --data_path weather.csv \
      --model_id weather_336_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --factor 1 \
      --des 'Exp' \
      --itr 1 \
      --gpu $gpu \
      --top_k 4 \
      --revin 0 \
      --learning_rate 0.0005 \
      --station_lr 0.00005 \
      --station_type $station_type >logs/LongForecasting/$model_name'_weather_'$pred_len'_'$station_type'_MSN'.log
    
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/ETT-small \
      --data_path ETTh2.csv \
      --model_id ETTh2_336_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --e_layers 3 \
      --n_heads 8 \
      --d_model 32 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --period_len 24 \
      --learning_rate 0.00005 \
      --station_lr 0.00005 \
      --revin 0 \
      --top_k 4 \
      --station_type $station_type >logs/LongForecasting/$model_name'_ETTh2_'$pred_len'_'$station_type'_MSN'.log
  done
done

for model_name in PatchTST; do
  for pred_len in 24 36 48 60; do
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/illness \
      --data_path national_illness.csv \
      --model_id ili_36_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 104 \
      --label_len 52 \
      --pred_len $pred_len \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --train_epochs 100\
      --patience 10\
      --station_type $station_type \
      --station_lr 0.0005 \
      --learning_rate 0.0005 \
      --itr 1 >logs/LongForecasting/$model_name'_ili_'$pred_len'_'$station_type'_MSN'.log
  done
done