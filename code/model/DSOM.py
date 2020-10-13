import sys
sys.path.append("..")
from config.config import cfg
import torch
import torch.nn as nn
import cv2
import os
from tensorboardX import SummaryWriter

class Flatten(nn.Module):
    def __init__(self, maintain = 1):

        super(Flatten, self).__init__()
        self.maintain = maintain

    def forward(self, x):
        
        N, C, _, _ = x.size() # read in N, C, H, W

        if self.maintain == 1:
            x = x.view(N, -1)
        elif self.maintain == 2:
            x = x.view(N, C, -1)

        return x


class sim_compute(nn.Module):
    def __init__(self):

        super(sim_compute, self).__init__()
        self.sim_cal = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, f_scan, f_map, zero_tensor, fmap_W, fmap_H):
        
        sim_map = zero_tensor.repeat(cfg.DSOM_batchsize,cfg.DSOM_K_portion,fmap_W,fmap_H)

        for i in range(cfg.DSOM_K_portion):

            f_scan_ = f_scan[:,:,i].squeeze(1).unsqueeze(-1).unsqueeze(-1)

            sim_map[:,i,:,:] = self.sim_cal(f_map,f_scan_)
        
        return sim_map

##############
#
#   Pose Generation Network: Take scan [W,H,1] and map [W,H,1] as inputs and output [W,H,K]
#
##############

class DSOM(torch.nn.Module):
    def __init__(self, D_in_scan , D_in_map):

        super(DSOM, self).__init__()

        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.maxpooling = nn.MaxPool2d(2)
        self.unpool = nn.MaxUnpool2d(2)
        self.relu = nn.ELU(inplace=True)
        
        self.conv1_1 = nn.Conv2d(D_in_map, 64, 3, padding = 1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_3 = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256)

        self.sim_conv = nn.Conv2d(cfg.DSOM_K_portion, 256, 3, padding=1)
        self.bn_sim_conv = nn.BatchNorm2d(256)

        channel_flattened = 16

        self.activation_scan2fmap = nn.ELU(inplace=True)
        
        self.scan2fmap = nn.Sequential(
            nn.Conv2d(D_in_map, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            self.activation_scan2fmap,
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            self.activation_scan2fmap,
            self.maxpooling,
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            self.activation_scan2fmap,
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            self.activation_scan2fmap,
            self.maxpooling,
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            self.activation_scan2fmap,
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            self.activation_scan2fmap,
            self.maxpooling,
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            self.activation_scan2fmap,
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            self.activation_scan2fmap,
            self.maxpooling,
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            self.activation_scan2fmap,
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            self.activation_scan2fmap,
            self.maxpooling,
            Flatten(2),
            nn.Linear(channel_flattened,channel_flattened*2),
            self.activation_scan2fmap,
            nn.Linear(channel_flattened*2,channel_flattened*2),
            self.activation_scan2fmap,
            nn.Linear(channel_flattened*2,cfg.DSOM_K_portion),
            self.activation_scan2fmap,
        )

        self.conv3_4_D = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_4_D = nn.BatchNorm2d(256)
        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3_D = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2_D = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.bn3_1_D = nn.BatchNorm2d(128)
        
        self.conv2_3_D = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_3_D = nn.BatchNorm2d(128)
        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2_D = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2_1_D = nn.BatchNorm2d(64)
        
        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2_D = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, cfg.DSOM_K_portion, 3, padding=1)
        self.bn1_1_D = nn.BatchNorm2d(cfg.DSOM_K_portion)

        self.sim_cal = sim_compute()

        '''
        self.conv4_1,
        self.conv4_2,
        self.conv4_3,
        
        self.conv5_1,
        self.conv5_2,
        self.conv5_3,
        
        self.conv5_3_D,
        self.conv5_2_D,
        self.conv5_1_D,
        
        self.conv4_3_D,
        self.conv4_2_D,
        self.conv4_1_D,
        '''

        self.finalize = nn.Sequential(nn.LogSoftmax(dim=-1))
       
    def forward(self,scan,map,zero_tensor):
        
        f_scan = self.scan2fmap(scan)
        # generate pmap

        pmap = map

        # Encoder block 1
        pmap = self.relu(self.bn1_1(self.conv1_1(pmap)))
        pmap = self.relu(self.bn1_2(self.conv1_2(pmap)))
        size1 = pmap.size()
        pmap, mask1 = self.pool(pmap)
        
        # Encoder block 2
        pmap = self.relu(self.bn2_1(self.conv2_1(pmap)))
        pmap = self.relu(self.bn2_2(self.conv2_2(pmap)))
        pmap = self.relu(self.bn2_3(self.conv2_3(pmap)))
        size2 = pmap.size()
        pmap, mask2 = self.pool(pmap)
        
        # Encoder block 3
        pmap = self.relu(self.bn3_1(self.conv3_1(pmap)))
        pmap = self.relu(self.bn3_2(self.conv3_2(pmap)))
        pmap = self.relu(self.bn3_3(self.conv3_3(pmap)))
        pmap = self.relu(self.bn3_4(self.conv3_4(pmap)))
        size3 = pmap.size()
        pmap, mask3 = self.pool(pmap)
        '''
        # Encoder block 4
        pmap = self.relu(self.conv4_1(pmap))
        pmap = self.relu(self.conv4_2(pmap))
        pmap = self.relu(self.conv4_3(pmap))
        size4 = pmap.size()
        pmap, mask4 = self.pool(pmap)
        
        # Encoder block 5
        pmap = self.relu(self.conv5_1(pmap))
        pmap = self.relu(self.conv5_2(pmap))
        pmap = self.relu(self.conv5_3(pmap))
        size5 = pmap.size()
        pmap, mask5 = self.pool(pmap)
        
        # Decoder block 5
        pmap = self.unpool(pmap, mask5, output_size = size5)
        pmap = self.relu(self.conv5_3_D(pmap))
        pmap = self.relu(self.conv5_2_D(pmap))
        pmap = self.relu(self.conv5_1_D(pmap))
        
        # Decoder block 4
        pmap = self.unpool(pmap, mask4, output_size = size4)
        pmap = self.relu(self.conv4_3_D(pmap))
        pmap = self.relu(self.conv4_2_D(pmap))
        pmap = self.relu(self.conv4_1_D(pmap))
        '''
        _ , _ , fmap_W, fmap_H = pmap.size()
        pmap = self.sim_cal(f_scan,pmap,zero_tensor,fmap_W,fmap_H)
        pmap = self.relu(self.bn_sim_conv(self.sim_conv(pmap)))

        # Decoder block 3
        pmap = self.unpool(pmap, mask3, output_size = size3)
        pmap = self.relu(self.bn3_4_D(self.conv3_4_D(pmap)))
        pmap = self.relu(self.bn3_3_D(self.conv3_3_D(pmap)))
        pmap = self.relu(self.bn3_2_D(self.conv3_2_D(pmap)))
        pmap = self.relu(self.bn3_1_D(self.conv3_1_D(pmap)))
        
        # Decoder block 2
        pmap = self.unpool(pmap, mask2, output_size = size2)
        pmap = self.relu(self.bn2_3_D(self.conv2_3_D(pmap)))
        pmap = self.relu(self.bn2_2_D(self.conv2_2_D(pmap)))
        pmap = self.relu(self.bn2_1_D(self.conv2_1_D(pmap)))

        # Decoder block 1
        pmap = self.unpool(pmap, mask1, output_size = size1)
        pmap = self.relu(self.bn1_2_D(self.conv1_2_D(pmap)))
        pmap = self.relu(self.bn1_1_D(self.conv1_1_D(pmap)))

        pmap = pmap.view(cfg.DSOM_batchsize,-1)
        
        return self.finalize(pmap)

