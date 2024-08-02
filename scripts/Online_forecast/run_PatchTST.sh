if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/Online_LongForecasting" ]; then
  mkdir ./logs/Online_LongForecasting
fi
gpu=0
station_type=adaptive
features=M
online_learning=stat_backbone

for model_name in PatchTST; do
  for pred_len in 96 192 336; do
    python -u run_longExp.py \
      --exp Exp_EvoMSN \
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
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --online_learning $online_learning \
      --batch_size 4 \
      --test_batch_size 1 \
      --itr 1 >logs/Online_LongForecasting/$model_name'_electricity_'$pred_len'_'$station_type'_EvoMSN'.log

    python -u run_longExp.py \
      --exp Exp_EvoMSN \
      --station_pretrain_epoch 5 \
      --is_training 1 \
      --root_path ./datasets/exchange_rate \
      --data_path exchange_rate.csv \
      --model_id exchange_336_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --online_learning $online_learning \
      --test_batch_size 1 \
      --itr 1 >logs/Online_LongForecasting/$model_name'_exchange_rate_'$pred_len'_'$station_type'_EvoMSN'.log

    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_EvoMSN \
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
      --des 'Exp' \
      --itr 1 \
      --gpu $gpu \
      --online_learning $online_learning \
      --batch_size 1 \
      --test_batch_size 1 \
      --station_type $station_type >logs/Online_LongForecasting/$model_name'_traffic_'$pred_len'_'$station_type'_EvoMSN'.log

    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_EvoMSN \
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
      --des 'Exp' \
      --itr 1 \
      --gpu $gpu \
      --online_learning $online_learning \
      --test_batch_size 1 \
      --station_type $station_type >logs/Online_LongForecasting/$model_name'_weather_'$pred_len'_'$station_type'_EvoMSN'.log

    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_EvoMSN \
      --root_path ./datasets/ETT-small \
      --data_path ETTh1.csv \
      --model_id ETTh1_336_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --online_learning $online_learning \
      --test_batch_size 1 >logs/Online_LongForecasting/$model_name'_ETTh1_'$pred_len'_'$station_type'_EvoMSN'.log
  
  python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_EvoMSN \
      --root_path  ./datasets/ETT-small \
      --data_path ETTm1.csv \
      --model_id ETTm1_336_$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --online_learning $online_learning \
      --test_batch_size 1 \
      --itr 1 >logs/Online_LongForecasting/$model_name'_Ettm1_'$pred_len'_'$online_learning'_'$station_type'_EvoMSN'.log
  done
done

