from argparse import ArgumentParser
import torch
from exp import exp_DWT
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
parser = ArgumentParser()
parser.add_argument('--data', type=str, default='custom', help='data of experiment, options: [ETTh1, ETTh2, ETTm1, ETTm2]') 
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of data')
parser.add_argument('--data_path', type=str, default='CC-PV.csv', help='data file')
parser.add_argument('--features', type=str, default='MS', help='features of experiment, options: [M, S, MS]')
parser.add_argument('--freq', type=str, default='t', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--seq_len', type=int, default=96, help='length of sequence')
parser.add_argument('--label_len', type=int, default=0, help='length of sequence')
parser.add_argument('--pred_len', type=int, default=24, help='length of prediction')
parser.add_argument('--c_in', type=int, default=8, help='encoder input size')
parser.add_argument('--c_out', type=int, default=8, help='encoder input size')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--batch_size', type=int, default=62, help='batch size of dataset')
parser.add_argument('--learning_rate', type=float, default=0.002489072819868877, help='learning rate') #0.001
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--dropout', type=float, default=0.09, help='dropout') #0.05
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--save_path', type=str, default='./checkpoints/', help='path to save checkpoints')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1',help='device ids of multile gpus')
parser.add_argument('--input', type=int, default=8, help='input dimension')
parser.add_argument('--down_sampling_method', type=str, default='avg', help='down_sampling_method')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--sampling_layers', type=int, default=2, help='samp layers')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--lradj', type=str, default='type3',help='adjust learning rate')
parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')

if __name__ == '__main__':
    args = parser.parse_args()
    exp = exp_DWT.Exp_model(args)
    settings = f"{args.data_path}"
    exp.train(settings)
    exp.test(settings)
    torch.cuda.empty_cache()


