if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi
seq_len=96
label_len=48
station_type=adaptive
features=M
gpu=0

for model_name in TimesNet; do
  for pred_len in 96 192 336; do
    python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/exchange_rate \
      --data_path exchange_rate.csv \
      --model_id exchange_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --d_model 64 \
      --d_ff 64 \
      --factor 3 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --itr 1 >logs/LongForecasting/$model_name'_exchange_rate_'$pred_len'_'$station_type'_MSN'.log

    python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/electricity \
      --data_path electricity.csv \
      --model_id electricity_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --d_model 256 \
      --d_ff 512 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --itr 1 >logs/LongForecasting/$model_name'_electricity_'$pred_len'_'$station_type'_MSN'.log

    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/traffic \
      --data_path traffic.csv \
      --model_id traffic_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --d_model 512 \
      --d_ff 512 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --itr 1 >logs/LongForecasting/$model_name'_traffic_'$pred_len'_'$station_type'_MSN'.log

    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/weather \
      --data_path weather.csv \
      --model_id weather_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --d_model 32 \
      --d_ff 32 \
      --des 'Exp' \
      --station_lr 0.00005 \
      --gpu $gpu \
      --station_type $station_type \
      --itr 1 >logs/LongForecasting/$model_name'_weather_'$pred_len'_'$station_type'_MSN'.log

    python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/ETT-small \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model 32 \
      --d_ff 32 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --itr 1 >logs/LongForecasting/$model_name'_Etth2_'$pred_len'_'$station_type'_MSN'.log
  done
done

for model_name in TimesNet; do
  for pred_len in 720; do
    python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/exchange_rate \
      --data_path exchange_rate.csv \
      --model_id exchange_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --batch_size 16 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --itr 1 >logs/LongForecasting/$model_name'_exchange_rate_'$pred_len'_'$station_type'_MSN'.log

    python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/electricity \
      --data_path electricity.csv \
      --model_id electricity_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --gpu $gpu \
      --batch_size 16 \
      --station_type $station_type \
      --itr 1 >logs/LongForecasting/$model_name'_electricity_'$pred_len'_'$station_type'_MSN'.log

    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/traffic \
      --data_path traffic.csv \
      --model_id traffic_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --gpu $gpu \
      --batch_size 16 \
      --station_type $station_type \
      --itr 1 >logs/LongForecasting/$model_name'_traffic_'$pred_len'_'$station_type'_MSN'.log

    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/weather \
      --data_path weather.csv \
      --model_id weather_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --gpu $gpu \
      --batch_size 16 \
      --station_lr 0.0001 \
      --learning_rate 0.0005 \
      --patience 5 \
      --station_type $station_type \
      --itr 1 >logs/LongForecasting/$model_name'_weather_'$pred_len'_'$station_type'_MSN'.log

    python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path ./datasets/ETT-small \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --batch_size 16 \
      --station_lr 0.0001 \
      --learning_rate 0.0005 \
      --patience 5 \
      --station_type $station_type \
      --itr 1 >logs/LongForecasting/$model_name'_Etth2_'$pred_len'_'$station_type'_MSN'.log
  done
done

for model_name in TimesNet; do
  for pred_len in 24 36 48; do
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path /home/user/data/illness \
      --data_path national_illness.csv \
      --model_id ili_36_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 36 \
      --label_len 18 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model 768 \
      --d_ff 768 \
      --des 'Exp' \
      --gpu $gpu \
      --learning_rate 0.0005 \
      --station_type $station_type \
      --itr 1 >logs/LongForecasting/$model_name'_ili_'$pred_len'_MSN'.log
  done
  for pred_len in 60; do
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_MSN \
      --root_path /home/user/data/illness \
      --data_path national_illness.csv \
      --model_id ili_36_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 36 \
      --label_len 18 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --station_lr 0.0001 \
      --learning_rate 0.0005 \
      --station_type $station_type \
      --itr 1 >logs/LongForecasting/$model_name'_ili_'$pred_len'_MSN'.log
  done
done
