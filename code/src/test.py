from config.config import cfg
import torch
from tensorboardX import SummaryWriter
from model.DSOM import DSOM, Flatten
from Data_Loaders.Data_Loader import DataLoader_new
import datetime

class test():
    def __init__(self,dataset,subdataset,batchsize,num_iters_per_epoch_val,device,load_model):
        self.device = device
        self.zero_tensor = torch.zeros(1, 1, 1, 1).to(device)
        self.batchsize = batchsize
        self.num_iters_per_epoch_val = num_iters_per_epoch_val
        self.DataLoader = DataLoader_new('../Dataset/', batchsize * 1, batchsize * num_iters_per_epoch_val, dataset, subdataset)

        print('Building model')
        self.model = DSOM(1, 1).to(device)
        print('Finish building model')

        if load_model is not None:
            print('Loading model...')
            check_point = torch.load(load_model,device)
            self.model.load_state_dict(check_point)
            print('Finish loading model')

        # Loss function
        self.loss = torch.nn.KLDivLoss(reduction='sum').to(device)
        self.Flatten = Flatten()

    def test(self):

            loss_val = 0
            self.model.eval()

            with torch.no_grad():
                for iter_val in range(self.num_iters_per_epoch_val):
                    map_batch_val, scan_batch_val, pmap_batch_val = self.DataLoader.get_Data(iter_val,
                                                                                        self.batchsize,
                                                                                        self.zero_tensor,
                                                                                        training=False)
                    pmap_pred_val = self.model(scan_batch_val, map_batch_val, self.zero_tensor)
                    pmap_gt_val = pmap_batch_val.view(cfg.DSOM_batchsize, -1)
                    Loss_val = self.loss(input=pmap_pred_val, target=pmap_gt_val)
                    loss_val += Loss_val.data

            print('Testing loss: '+str(loss_val / (self.num_iters_per_epoch_val * self.batchsize)))