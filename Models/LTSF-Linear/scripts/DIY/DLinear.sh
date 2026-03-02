#!/bin/bash

if [ ! -d "./logs" ]; then
	mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
	mkdir ./logs/LongForecasting
fi

seq_len=336
model_name=DLinear

# 1. Set default values (optional)
DATA=""
PRED_LEN=0

# 2. Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data)
      DATA="$2"
      shift 2 # Move past the flag and its value
      ;;
    --pred_len)
      PRED_LEN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

cd ../..
touch logs/LongForecasting/$model_name'_'DIY_$seq_len'_'$PRED_LEN.log

python -u run_longExp.py \
	--is_training 1 \
	--root_path ./dataset \
	--data_path $DATA \
	--model_id DLinearDIY_$seq_len'_'$PRED_LEN \
	--model DLinear \
	--data custom \
	--features M \
	--seq_len $seq_len \
	--pred_len $PRED_LEN \
	--enc_in 8 \
	--des 'Exp' \
	--itr 1 --batch_size 8 --learning_rate 0.0005 > logs/LongForecasting/$model_name'_'DIY_$seq_len'_'$PRED_LEN.log

