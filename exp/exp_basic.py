import os
import torch
import numpy as np

class Exp_Basic(object):
    # I highlited this to deal messing device assignment errors
    # def __init__(self, args):
    #     self.args = args
    #     # self.device = self._acquire_device()
    #     try:
    #         self.model = self._build_model().to(args.rank)
    #     except:
    #         self.model = self._build_model().cuda()
    def __init__(self, args):
        self.args = args
        #choose devise safely and move there
        gpu_idx = getattr(self.args, 'gpu', None)
        if gpu_idx is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_idx}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)
        # build and move model to chosen device
        #self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    # def _acquire_device(self):
    #     if self.args.use_gpu:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
    #         device = torch.device('cuda:{}'.format(self.args.gpu))
    #         print('Use GPU: cuda:{}'.format(self.args.gpu))
    #     else:
    #         device = torch.device('cpu')
    #         print('Use CPU')
    #     return device

    def _get_data(self):
        pass

    def valid(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    