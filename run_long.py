#from exp.exp_ETT import Exp_ETT
from exp.exp_ETT_new import Exp_ETT
import argparse
import torch
import numpy as np

# Argument parser setup
parser = argparse.ArgumentParser()

# Model settings
parser.add_argument('--model', type=str, default='SpikW', help='model of the experiment')

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
parser.add_argument('--keep_levels', type=str, default='all', help='Haar levels to keep: "all", "detail", "approx", or comma-separated ints like "0,1"')
parser.add_argument('--wavelet', type=str, default='db4', choices=['db4', 'db8', 'sym4', 'sym8'], help='Wavelet type')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads for SDSA')  
parser.add_argument('--use_gray_pe', type=int, default=1, help='Use Gray-PE (0=off, 1=on)')
parser.add_argument('--use_log_pe', type=int, default=1, help='Use Log-PE (0=off, 1=on)')
parser.add_argument('--gray_num_bits', type=int, default=10, help='Number of Gray code bits')
parser.add_argument('--dwc_kernel', type=int, default=5, help='DWC kernel size (3, 5, or 7)')


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
parser.add_argument('--tau', type=float, default=2.0)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--save', action='store_true', default=False, help='saving result')


parser.add_argument('--J', type=int, default=5)

# FGN
parser.add_argument('--seq_length', type=int, default=12, help='inout length')
parser.add_argument('--pre_length', type=int, default=12, help='predict length')
parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions')
parser.add_argument('--proj_dim', type=int, default=16, help='proj dim')
parser.add_argument('--feature_size', type=int, default='140', help='feature size')


args = parser.parse_args()



def main(rank):
    mse_list = []
    mae_list = []
    seed_list = [2022, 2023, 2024]

    num_runs = 3 # change it to 5 when we will do 5 runs.
    # NOTE: it is better to put seed numbers as year 2022, 2023, 2024/ for five you can change to 2025, 2026 by adding.

    for run in range(num_runs):
        print(f"\n================ Run {run + 1}/{num_runs} ================\n")

        seed = seed_list[run] # fix this to make seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

        # Initialize experiment
        Exp = Exp_ETT

        setting = '{}_{}_ft{}_sl{}_pl{}_lr{}_bs{}_run{}'.format(
            args.model, args.data, args.features, args.seq_len,
            args.pred_len, args.lr, args.batch_size, run
        )

        args.rank = rank
        exp = Exp(args)

        # Training
        print('>>>>>>> start training : {} >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        # Testing
        print('>>>>>>> testing : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mse, mae = exp.test(setting)

        print('Run {} | MSE: {:.4f}, MAE: {:.4f}'.format(run + 1, mse, mae))

        mse_list.append(mse)
        mae_list.append(mae)

    # ===== Final averages =====
    avg_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)
    avg_mae = np.mean(mae_list)
    std_mae = np.std(mae_list)

    print("\n================ Final Results ================")
    print("Average MSE: {:.4f} ± {:.4f}".format(avg_mse, std_mse))
    print("Average MAE: {:.4f} ± {:.4f}".format(avg_mae, std_mae))


if __name__ == '__main__':
    main(args.rank)
