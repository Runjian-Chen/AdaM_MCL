import sys
sys.path.append("..")
from config.config import cfg
import torch
from tensorboardX import SummaryWriter
from model.DSOM import DSOM, Flatten
from Data_Loaders.Data_Loader import DataLoader_new
import datetime

class train():
    def __init__(self,dataset,subdataset,batchsize,num_iters_per_epoch_train,num_iters_per_epoch_val,num_epochs,device,load_model,save_per_x_epoch,decay_per_x_step,decay_rate,val_per_x_iters):
        self.device = device
        self.zero_tensor = torch.zeros(1, 1, 1, 1).to(device)
        self.batchsize = batchsize
        self.num_iters_per_epoch_train = num_iters_per_epoch_train
        self.num_iters_per_epoch_val = num_iters_per_epoch_val
        self.num_epochs = num_epochs
        self.DataLoader = DataLoader_new('../Dataset/', batchsize * num_iters_per_epoch_train, batchsize * num_iters_per_epoch_val, dataset, subdataset)

        print('Building model')
        self.model = DSOM(1, 1).to(device)
        print('Finish building model')

        if load_model is not None:
            print('Loading model...')
            check_point = torch.load(load_model,device)
            self.model.load_state_dict(check_point)
            print('Finish loading model')

        self.log_path = '../Trained_models/' + dataset + '_' + subdataset + '_' + str(datetime.datetime.now()) + '/'

        # Log writer
        self.writer = SummaryWriter(self.log_path+'log')

        # Loss function
        self.loss = torch.nn.KLDivLoss(reduction='sum').to(device)
        self.Flatten = Flatten()

        # Optimizer
        self.l_rate = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.l_rate, weight_decay=0.000001)

        # Global step accumulator
        self.global_step = 1

        # Save model information
        with open(self.log_path+'model.txt', "w") as text_file:
            text_file.write(str(self.model))

        self.save_per_x_epoch = save_per_x_epoch
        self.decay_per_x_step = decay_per_x_step
        self.decay_rate = decay_rate
        self.val_per_x_iters = val_per_x_iters

    def train(self):
        # Training
        for epoch in range(self.num_epochs):

            if epoch % self.save_per_x_epoch == 0:
                torch.save(self.model.state_dict(), self.log_path+'model_train_'+str(epoch)+'epochs.pth')

            # training
            for iter in range(self.num_iters_per_epoch_train):

                map_batch, scan_batch, pmap_batch = self.DataLoader.get_Data(iter, self.batchsize, self.zero_tensor)

                pmap_pred = self.model(scan_batch, map_batch, self.zero_tensor)

                pmap_gt = pmap_batch.view(self.batchsize, -1)

                Loss = self.loss(input=pmap_pred, target=pmap_gt) / self.batchsize

                self.optimizer.zero_grad()

                Loss.backward()

                self.optimizer.step()

                print('Learning rate: ' + str(self.l_rate))
                print('[%s %d: %d/%d] %s loss: %f' % (
                    'Probability Map Generation', epoch, iter, self.num_iters_per_epoch_train - 1, cfg.green('train'),
                    Loss.data))
                self.writer.add_scalars('./log/Train_loss', {'.log/train_loss': Loss.data},
                                   iter + epoch * self.num_iters_per_epoch_train)

                self.global_step = self.global_step + 1

                if (self.global_step % self.decay_per_x_step == 0):
                    self.l_rate *= self.decay_rate
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.l_rate, weight_decay=0.000001)

                if (iter + 1) % self.val_per_x_iters == 0:

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

                    self.model.train()
                    print('[%s %d: %d/%d] %s loss: %f' % (
                        'Probability Map Generation', epoch, iter, self.num_iters_per_epoch_train - 1, cfg.blue('validaton'),
                        loss_val / (self.num_iters_per_epoch_val * self.batchsize)))
                    self.writer.add_scalars('.log/Validation_loss',
                                       {'.log/val_loss': loss_val / (self.num_iters_per_epoch_val * self.batchsize)},
                                       iter + epoch * self.num_iters_per_epoch_train)