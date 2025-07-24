import optuna
from argparse import ArgumentParser
import argparse
import torch
from models.model import mymodel
from exp import exp_myself
import numpy as np
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# parser = ArgumentParser()
# parser.add_argument('--model', type=str, default='my', help='model of experiment, options: [timese]')
# parser.add_argument('--data', type=str, default='custom', help='data of experiment, options: [ETTh1, ETTh2, ETTm1, ETTm2]') 
# # parser.add_argument('--root_path', type=str, default='/home/gpuadmin/qihuifile/Time-Series-Library-main/data/ETT/', help='root path of data')
# parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of data')
# parser.add_argument('--data_path', type=str, default='XJ.csv', help='data file')
# parser.add_argument('--features', type=str, default='MS', help='features of experiment, options: [M, S, MS]')
# parser.add_argument('--freq', type=str, default='t', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
# parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
# parser.add_argument('--seq_len', type=int, default=96, help='length of sequence')
# parser.add_argument('--label_len', type=int, default=0, help='length of sequence')
# parser.add_argument('--pred_len', type=int, default=24, help='length of prediction')
# parser.add_argument('--c_in', type=int, default=8, help='encoder input size')
# parser.add_argument('--c_out', type=int, default=8, help='encoder input size')
# parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
# parser.add_argument('--batch_size', type=int, default=256, help='batch size of dataset')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
# parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
# parser.add_argument('--dropout', type=float, default=0.05, help='dropout') #0.05
# parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
# parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
# parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
# parser.add_argument('--save_path', type=str, default='./checkpoints/', help='path to save checkpoints')
# parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
# parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
# parser.add_argument('--gpu', type=int, default=1, help='gpu')
# parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
# parser.add_argument('--devices', type=str, default='0,1',help='device ids of multile gpus')
# parser.add_argument('--input', type=int, default=8, help='input dimension')
# parser.add_argument('--down_sampling_method', type=str, default='avg', help='down_sampling_method')
# parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
# parser.add_argument('--sampling_layers', type=int, default=1, help='samp layers')
# parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
# parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
# parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
# parser.add_argument('--lradj', type=str, default='type3',help='adjust learning rate')
# parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')

def object(trial):
    # learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    # learning_rate = trial.suggest_discrete_uniform("learning_rate", 1e-5, 1e-1,1e-2)
    batch_size_ls = [8,16,32,64,128]
    batch_size = trial.suggest_categorical("batch_size", batch_size_ls)
    d_modellist = [64,128,256,512,1024]
    d_model = trial.suggest_categorical("d_model", d_modellist)
    d_ff = trial.suggest_int("d_ff", 128, 2048)
    # dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
    dropout = trial.suggest_discrete_uniform("dropout", 0.0, 0.5,0.1)
    # seq_len = trial.suggest_int("seq_len", 24, 192)
    # pred_len = trial.suggest_int("pred_len", 12, 48)
    fixed_args = {
        "data": "custom",
        "root_path": "./dataset/",
        "data_path": "XJ.csv",
        "features": "MS",
        "freq": "t",
        "target": "OT",
        "c_in": 8,
        "c_out": 8,
        "embed": "timeF",
        "num_workers": 10,
        "train_epochs": 50,
        "patience": 3,
        "save_path": "./checkpoints/",
        "use_amp": False,
        "use_gpu": True,
        "gpu": 1,
        "use_multi_gpu": False,
        "devices": "0,1",
        "input": 8,
        "down_sampling_method": "avg",
        "use_norm": 1,
        "sampling_layers": 2,
        "inverse": False,
        "cols": None,
        "checkpoints": "./checkpoints/",
        "lradj": "type3",
        "seq_len":96,
        "label_len":0,
        "pred_len":24,
        "learning_rate":1e-2
    }
    dynamic_args = {
        # "learning_rate": learning_rate,
        "batch_size": batch_size,
        "d_model": d_model,
        "d_ff": d_ff,
        "dropout": dropout,
    }
    args = argparse.Namespace(**{**fixed_args, **dynamic_args})
    settings = f"{args.seq_len}_{args.pred_len}_{args.use_gpu}_{args.data_path}"
    exp = exp_myself.Exp_model(args)
    exp.train(settings)
    mae = exp.test(settings)
    # exp.my_Model_EXP(settings)
    return mae


if __name__ == '__main__':
    # args = parser.parse_args()
    # exp = exp_myself.Exp_model(args)
    # settings = f"{args.seq_len}_{args.pred_len}_{args.use_gpu}_{args.data_path}"
    # exp.train(settings)
    # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(settings))
    # exp.test(settings)
    # exp.my_Model_EXP(settings)
    # wandb.finish()
    # if args.do_predict:
    #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #     exp.predict(setting, True)
    study = optuna.create_study(direction="minimize")
    study.optimize(object, n_trials=50)

    # 输出最佳超参数和最优值
    print("Best trial:")
    trial_best = study.best_params
    print(trial_best)
    np.save("./sensitivity_analysis/bestpar.npy", trial_best)

    all_expert = study.trials
    np.save('./sensitivity_analysis/allexper.npy', all_expert)
    torch.cuda.empty_cache()
