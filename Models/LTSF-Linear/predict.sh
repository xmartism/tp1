
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
IS_TRAINING=1

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
    --is_training)
      IS_TRAINING="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

DATA_TRIMMED=${DATA%.csv}


# CONDA ENV
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

ENV_NAME="LTSF_Linear"

if ! conda info --envs | grep -q "$ENV_NAME"; then
	echo "ENVIROMENT $ENV_NAME not found. Creating..."
	conda create -n "$ENV_NAME" python=3.6.9
fi


# Samotny program na spustenie
conda run -n "$ENV_NAME" python -u run_longExp.py \
  --is_training $IS_TRAINING \
  --do_predict \
  --root_path ./dataset/ \
  --data_path $DATA \
  --model_id $DATA_TRIMMED'_'_$seq_len'_'$PRED_LEN \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $PRED_LEN \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0005  >logs/LongForecasting/$model_name'_'$DATA_TRIMMED'_'$seq_len'_'$PRED_LEN.log 

