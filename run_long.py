from exp.exp_ETT import Exp_ETT
import argparse
import torch

# Argument parser setup
parser = argparse.ArgumentParser()

# Model settings
parser.add_argument('--model', type=str, default='SpikF', help='model of the experiment')

# Dataset settings
parser.add_argument('--data', type=str, required=False, default='ETTh1', 
                    choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'traffic', 'weather', 
                             'electricity', 'metr-la', 'pems-bay', 'solar-energy', 'exchange'], 
                    help='name of dataset')
parser.add_argument('--root_path', type=str, default='./datasets/long/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='location of the data file')
parser.add_argument('--features', type=str, default='M', choices=['S', 'M'], 
                    help='features S is univariate, M is multivariate')
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--checkpoints', type=str, default='exp/run_ETT/', help='location of model checkpoints')

# Model settings
parser.add_argument('--seq_len', type=int, default=96, help='look back window')
parser.add_argument('--pred_len', type=int, default=336, help='prediction sequence length, horizon')
parser.add_argument('--levels', type=int, default=2)
parser.add_argument('--T', type=int, default=16)
parser.add_argument('--patch_num', type=int, default=48)
parser.add_argument('--patch_dim', type=int, default=32)
parser.add_argument('--alpha', type=float, default=2.0)
parser.add_argument('--hidden_dim', type=int, default=720)
parser.add_argument('--random_seed', type=int, default=0)

# Training settings
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=0, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mae', help='loss function')
parser.add_argument('--model_name', type=str, default='SpikF')
parser.add_argument('--evaluate', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)

args = parser.parse_args()

def main(rank):
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # Initialize experiment
    Exp = Exp_ETT

    setting = '{}_{}_ft{}_sl{}_pl{}_imp{}_lr{}_bs{}_eh{}_dh{}_l{}_itr0_K{}'.format(
        args.model, args.data, args.features, args.seq_len, args.pred_len, args.K_IMP, 
        args.lr, args.batch_size, args.enc_hidden, args.dec_hidden, args.levels, args.K_IMP)
    
    args.rank = rank
    exp = Exp(args)
    torch.cuda.set_device(1)

    # Start training
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    # Start testing
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mse, mae = exp.test(setting)
    print('Final mean normed mse:{:.4f}, mae:{:.4f}'.format(mse, mae))

if __name__ == '__main__':
    main(args.rank)