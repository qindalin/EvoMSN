if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/Online_LongForecasting" ]; then
  mkdir ./logs/Online_LongForecasting
fi
seq_len=96
label_len=48
station_type=adaptive
features=M
gpu=0
online_learning=stat_backbone

for model_name in TCN; do
  for pred_len in 96 192 336; do
    python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_EvoMSN \
      --root_path ../data/exchange_rate \
      --data_path exchange_rate.csv \
      --model_id exchange_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --online_learning $online_learning \
      --test_batch_size 1 \
      --itr 1 >logs/Online_LongForecasting/$model_name'_exchange_rate_'$pred_len'_'$online_learning'_'$station_type'_EvoMSN'.log

    python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_EvoMSN \
      --root_path ../data/electricity \
      --data_path electricity.csv \
      --model_id electricity_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --online_learning $online_learning \
      --test_batch_size 1 \
      --itr 1 >logs/Online_LongForecasting/$model_name'_electricity_'$pred_len'_'$online_learning'_'$station_type'_EvoMSN'.log

    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_EvoMSN \
      --root_path ../data/traffic \
      --data_path traffic.csv \
      --model_id traffic_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --online_learning $online_learning \
      --test_batch_size 1 \
      --itr 1 >logs/Online_LongForecasting/$model_name'_traffic_'$pred_len'_'$online_learning'_'$station_type'_EvoMSN'.log

    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_EvoMSN \
      --root_path ../data/weather \
      --data_path weather.csv \
      --model_id weather_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --online_learning $online_learning \
      --test_batch_size 1 \
      --itr 1 >logs/Online_LongForecasting/$model_name'_weather_'$pred_len'_'$online_learning'_'$station_type'_EvoMSN'.log

    python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_EvoMSN \
      --root_path ../data/ETT-small \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --online_learning $online_learning \
      --test_batch_size 1 \
      --itr 1 >logs/Online_LongForecasting/$model_name'_Etth1_'$pred_len'_'$online_learning'_'$station_type'_EvoMSN'.log
  
  python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_EvoMSN \
      --root_path  /home/user/data/ETT-small \
      --data_path ETTm1.csv \
      --model_id ETTm1_96_$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features $features \
      --seq_len 96 \
      --label_len 48 \
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