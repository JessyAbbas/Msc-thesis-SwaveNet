import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import warnings

from models.SpikF import SpikF
# from model.spike_tcn import STCN
# from model.spikernn import SRNN
# from model.spikformer import Spikformer
# from model.QKformer import QKFormer
from timm.models import create_model

from utils.metrics import metric, metric_
from utils.tools import EarlyStopping
warnings.filterwarnings('ignore')
from data_provider.ETT_data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from exp.exp_basic import Exp_Basic
from spikingjelly.clock_driven import functional
torch.autograd.set_detect_anomaly(True)
from torch.optim.lr_scheduler import CosineAnnealingLR

class Exp_ETT(Exp_Basic):
    def __init__(self, args):
        super(Exp_ETT, self).__init__(args)
        self.test_loader = self._get_data(flag = 'test')
        self.shuffle_indexs = torch.tensor([18, 49, 62, 77, 51, 50, 35, 13, 32, 87, 54, 61, 15, 45, 25, 21,  6, 58,
        93, 89,  1, 92, 83, 26, 47,  0, 78,  4, 38, 24, 16, 17,  8, 20, 27, 67,
        53, 82, 34, 10, 74, 44, 52, 43, 41, 30, 88, 84, 59,  3, 39, 57,  7, 68,
        23, 79, 81, 66, 76, 31, 63, 86, 28, 71, 56,  9, 90, 48, 14, 60, 94, 64,
        12, 91,  5, 37, 29, 95,  2, 65, 80, 73, 55, 22, 11, 72, 75, 69, 85, 42,
        46, 40, 70, 33, 36, 19])
        

    def _build_model(self):
        if self.args.features == 'S':
            self.input_dim = 1
        elif self.args.features == 'M':
            if "ETT" in self.args.data:
                self.input_dim = 7
            elif self.args.data == 'ECL' or self.args.data == 'electricity':
                self.input_dim = 321
            elif self.args.data == 'solar_AL':
                self.input_dim = 137
            elif self.args.data == 'exchange':
                self.input_dim = 8
            elif self.args.data == 'traffic':
                self.input_dim = 862
            elif self.args.data == 'weather':
                self.input_dim = 21
            elif self.args.data == 'illness':
                self.input_dim = 7
            elif self.args.data == 'metr-la':
                self.input_dim= 207
            elif self.args.data == 'pems-bay':
                self.input_dim = 325
            elif self.args.data == 'solar-energy':
                self.input_dim=137
            elif self.args.data== 'exchange':
                self.input_dim = 8
        else:
            print('Error!')

        if self.args.model == 'QKFormer': 
            model = create_model(
                "QKFormer",
                pretrained=False,
                drop_rate=0.,
                drop_path_rate=0.2,
                drop_block_rate=None,
                img_size_h=self.args.seq_len, img_size_w=self.input_dim,
                patch_size=4, embed_dims=384, num_heads=8, mlp_ratios=4,
                in_channels=1, num_classes=self.input_dim*self.args.pred_len, qkv_bias=False,
                depths=self.args.levels, sr_ratios=1,
                T=self.args.T,
                pred_len=self.args.pred_len
            )
        elif self.args.model == 'STCN':
            model = STCN(
                self.args.seq_len,
                self.args.patch_num,
                self.args.patch_dim,
                self.args.T,
                self.args.levels,
                self.input_dim,
                self.args.pred_len,
                self.args.tau,
                self.args.alpha,
                self.args.hidden_dim,
                self.args.mean,
                self.args.last,
                self.args.std
            )
        elif self.args.model=='SRNN':
            model=SRNN(
                self.args.seq_len,
                self.args.patch_num,
                self.args.patch_dim,
                self.args.T,
                self.args.levels,
                self.input_dim,
                self.args.pred_len,
                self.args.tau,
                self.args.alpha,
                self.args.hidden_dim,
                self.args.mean,
                self.args.last,
                self.args.std
            )
        elif self.args.model=='Spikformer':
            model=Spikformer(
                self.args.seq_len,
                self.args.patch_num,
                self.args.patch_dim,
                self.args.T,
                self.args.levels,
                self.input_dim,
                self.args.pred_len,
                self.args.tau,
                self.args.alpha,
                self.args.hidden_dim,
                self.args.mean,
                self.args.last,
                self.args.std
            )
        else:
            model = SpikF(
                self.args.seq_len,
                self.args.patch_num,
                self.args.patch_dim,
                self.args.T,
                self.args.levels,
                self.input_dim,
                self.args.pred_len,
                self.args.tau,
                self.args.alpha,
                self.args.hidden_dim
            )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'weather':Dataset_Custom,
            'ECL':Dataset_Custom,
            'electricity':Dataset_Custom,
            'Solar':Dataset_Custom,
            'traffic':Dataset_Custom,
            'exchange':Dataset_Custom,
            'illness':Dataset_Custom,
            'exchange': Dataset_Custom
        }
        Data = data_dict[self.args.data]

        if flag == 'test' or flag == 'val':
            shuffle_flag = False; drop_last = False; batch_size = 1
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size 
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag, 
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
            )

        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999))
        return model_optim

    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        # criterion = nn.HuberLoss(reduction='mean', delta=0.3)
        return criterion

    def train(self, setting):
        torch.cuda.empty_cache()
        # torch.autograd.set_detect_anomaly(True)
        train_loader = self._get_data(flag = 'train')
        valid_loader = self._get_data(flag = 'val')
        self.test_loader = self._get_data(flag = 'test')
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=0.0002)
        # ema = EMA(self.model, decay=0.9)
        model_optim = self._select_optimizer()
        scheduler = CosineAnnealingLR(model_optim, T_max=20)
        criterion =  self._select_criterion(self.args.loss)

        # if self.args.use_amp:
        #     scaler = torch.cuda.amp.GradScaler()

        epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                # batch_x = batch_x[:, self.shuffle_indexs, :]
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch_DPAD(batch_x, batch_y)
                # print(pred)
                if len(pred.shape)==4:
                    true = true.repeat(self.args.T, 1, 1, 1)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i+1) % 100==0:
                    avg_loss = sum(train_loss[-100:]) / 100
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, avg_loss))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.7f}s/iter; left time: {:.7f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # if self.args.use_amp:
                #     print('use amp')
                #     scaler.scale(loss).backward()
                #     scaler.step(model_optim)
                #     scaler.update()
                else:
                    loss.backward(create_graph=True)
                    model_optim.step()
                    
                    

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            print('--------start to validate-----------')
            valid_loss = self.valid(valid_loader, criterion, flag="valid")
            print('--------start to test-----------')
            test_loss = self.valid(self.test_loader, criterion, flag="test")

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss, test_loss))

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, test_loss))


            # ema.update(self.model)
            scheduler.step()
            early_stopping(valid_loss, self.model, path)
            # early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # lr = adjust_learning_rate(model_optim, epoch+1, self.args)
        lr=0
        
        best_model_path = path+'/'+'checkpoint.pth'
        
        return best_model_path

    def valid(self, valid_loader, criterion, flag):
        torch.cuda.empty_cache()
        self.model.eval()
        total_loss = []

        mses = []
        maes = []
        weight = []


        for i, (batch_x, batch_y) in enumerate(valid_loader):
            # batch_x = batch_x[:, self.shuffle_indexs, :]
            pred, true = self._process_one_batch_DPAD(batch_x, batch_y)
            weight.append(true.shape[0]/self.args.batch_size)
            if len(pred.shape) ==4:
                pred = pred.mean(dim=0)

            mae, mse = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            mses.append(mse)
            maes.append(mae)

            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)


        total_loss = np.average(total_loss)
        mse = sum(w * m for w, m in zip(weight, mses)) / sum(weight)
        mae = sum(w * m for w, m in zip(weight, maes)) / sum(weight)


        print('-----------start to {} {}-----------\n|  Normed  | mse:{:5.7f} | mae:{:5.7f} |'.format(self.args.rank, flag, mse, mae))
        
        return total_loss


    def test(self, setting, evaluate=0):
        torch.cuda.empty_cache()
        self.model.eval()
        
        preds = []
        trues = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        for i, (batch_x, batch_y) in enumerate(self.test_loader):
            # batch_x = batch_x[:, self.shuffle_indexs, :]
            pred, true = self._process_one_batch_DPAD(batch_x, batch_y)
            # print(pred)
            if len(pred.shape) ==4:
                pred = pred.mean(dim=0)
            pred = pred.detach().cpu()
            true = true.detach().cpu()

            preds.append(pred)
            trues.append(true)


        preds = torch.cat(preds, dim=0).numpy()
        print(preds.shape)
        trues = torch.cat(trues, dim=0).numpy()

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])


        mae, mse, RSE, R2 = metric_(preds, trues)
        print('|  Normed  | mse:{:5.7f} | mae:{:5.7f} | RSE:{:5.7f} | R^2:{:5.7f} |'.format(mse, mae, RSE, R2))
                
        # result save
        if self.args.save:
            folder_path = 'exp/ETT_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, RSE, R2 = metric_(preds, trues)
            print('|  Normed  | mse:{:5.7f} | mae:{:5.7f} | RSE:{:5.7f} | R^2:{:5.7f} |'.format(mse, mae, RSE, R2))
      

        return mse, mae 


    def _process_one_batch_DPAD(self,batch_x, batch_y):
        batch_x = batch_x.float().to(self.args.rank)
        batch_y = batch_y.float()
        functional.reset_net(self.model)
        outputs = self.model(batch_x)

        f_dim = -1 if self.args.features=='MS' else 0
        # batch_y
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.args.rank)

        return outputs, batch_y


        
