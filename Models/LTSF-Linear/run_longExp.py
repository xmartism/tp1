import argparse
import os
import random

import numpy as np
import torch

from exp.exp_main import Exp_Main


TRAIN_MODE = 'train'
PREDICT_MODE = 'predict'


def _validate_args(args):
    if args.lookback_window is None:
        raise ValueError('--lookback-window is required.')

    if args.mode == TRAIN_MODE:
        if not args.train_dataset:
            raise ValueError('--train-dataset is required in --mode train.')
        if not args.val_dataset:
            raise ValueError('--val-dataset is required in --mode train.')
    elif args.mode == PREDICT_MODE:
        if not args.context_dataset:
            raise ValueError('--context-dataset is required in --mode predict.')

    if not args.model_dir:
        raise ValueError('--model-dir is required.')


def _build_setting(args):
    return '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.horizon,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Autoformer & Transformer family for Time Series Forecasting'
    )

    # basic config
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lookback-window', type=int)
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument(
        '--train_only',
        type=bool,
        required=False,
        default=False,
        help='perform training on full input dataset without validation and testing',
    )
    parser.add_argument('--model_id', type=str, required=False, default='DLinear_custom', help='model id')
    parser.add_argument(
        '--model',
        type=str,
        required=False,
        default='DLinear',
        help='model name, options: [Autoformer, Informer, Transformer]',
    )
    parser.add_argument('--train-dataset', required=False)
    parser.add_argument('--val-dataset', required=False)
    parser.add_argument('--test-dataset', required=False)
    parser.add_argument('--context-dataset', required=False)
    parser.add_argument('--model-dir', type=str, required=False, default='./model', help='directory for trained model artifacts')
    parser.add_argument('--output', type=str, required=False, default='results.csv', help='file for prediction output')
    parser.add_argument('--mode', type=str, choices=[TRAIN_MODE, PREDICT_MODE], default=TRAIN_MODE)

    # data loader
    parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, required=False, default='./', help='root path of the data file')
    parser.add_argument('--dataset', type=str, required=False, default='', help='data file')
    parser.add_argument(
        '--features',
        required=False,
        type=str,
        default='MS',
        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate',
    )
    parser.add_argument('--target', required=False, type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument(
        '--freq',
        required=False,
        type=str,
        default='b',
        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h',
    )
    parser.add_argument('--checkpoints', required=False, type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--date', default='date', required=False, help='date column name')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--horizon', required=True, type=int, default=96, help='prediction sequence length')

    # DLinear
    parser.add_argument(
        '--individual',
        required=False,
        action='store_true',
        default=False,
        help='DLinear: a linear layer for each variate(channel) individually',
    )

    # Formers
    parser.add_argument(
        '--embed_type',
        type=int,
        default=0,
        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding',
    )
    parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument(
        '--distil',
        action='store_false',
        help='whether to use distilling in encoder, using this argument means not using distilling',
        default=True,
    )
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()
    _validate_args(args)

    args.do_predict = args.mode == PREDICT_MODE
    args.is_training = 1 if args.mode == TRAIN_MODE else 0
    args.individual = True
    args.seq_len = args.lookback_window
    args.label_len = min(args.label_len, args.seq_len)
    args.checkpoints = args.model_dir

    os.makedirs(args.model_dir, exist_ok=True)

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    exp = Exp_Main(args)
    setting = _build_setting(args)

    if args.mode == TRAIN_MODE:
        exp.train(setting)
    else:
        print('>>>>>>>predicting<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.predict(setting, load=True)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
