import sys
sys.path.append("..")
from config.config import cfg
from torch.utils.data import Dataset
import torch
import numpy as np
import math
from src.utils import img_loader, img_loader_scan, txt_loader

class DataLoader_new(Dataset):

    def __init__(self, root_dir, tr_num, v_num, dataset, subdataset):
        self.fixed_map_train = img_loader(root_dir,
                                   dataset + '/Data_Transformed' + '/' + subdataset + '/train/map/')
        self.scan_info_train = img_loader_scan(root_dir,
                                         dataset + '/Data_Transformed'  + '/' + subdataset + '/train/scan/')
        self.pmap_info_train = img_loader(root_dir,
                                    dataset + '/Data_Transformed' + '/' + subdataset + '/train/pmap/')
        self.rotation_info_train = txt_loader(root_dir,
                                        dataset + '/Data_Transformed' + '/' + subdataset + '/train/rotation/')
        num_data = self.fixed_map_train.__len__()
        self.index = np.arange(num_data)
        np.random.shuffle(self.index)
        self.training_index = self.index[0:tr_num]

        self.fixed_map_val = img_loader(root_dir,
                                          dataset + '/Data_Transformed' + '/' + subdataset + '/val/map/')
        self.scan_info_val = img_loader_scan(root_dir,
                                               dataset + '/Data_Transformed' + '/' + subdataset + '/val/scan/')
        self.pmap_info_val = img_loader(root_dir,
                                          dataset + '/Data_Transformed' + '/' + subdataset + '/val/pmap/')
        self.rotation_info_val = txt_loader(root_dir,
                                              dataset + '/Data_Transformed' + '/' + subdataset + '/val/rotation/')
        num_data = self.fixed_map_val.__len__()
        self.index = np.arange(num_data)
        np.random.shuffle(self.index)
        self.validation_index = self.index[0:v_num]


        self.K_degree = torch.linspace(cfg.rad_min, cfg.rad_max, cfg.PGN_K_portion)
        self.subdataset = subdataset
        #self.X = cfg.Bicocca_crop[int(self.subdataset) - 1][2] - cfg.Bicocca_crop[int(self.subdataset) - 1][0]
        #self.Y = cfg.Bicocca_crop[int(self.subdataset) - 1][3] - cfg.Bicocca_crop[int(self.subdataset) - 1][1]
        #self.scale_factor = math.sqrt(cfg.PGN_X * cfg.PGN_Y / (self.X * self.Y))
        self.scaled_X = 128
        self.scaled_Y = 128

        print('Data summary:')
        print('Total: ' + str(tr_num + v_num))
        print('Training samples: ' + str(tr_num))
        print('Validating samples: ' + str(v_num))

    def get_Data(self, iter, batchsize, zero_tensor, training=True):

        scaled_X = self.scaled_X
        scaled_Y = self.scaled_Y

        data_fixed_map = zero_tensor.repeat(batchsize, 1, scaled_X, scaled_Y)
        data_scaninfo = zero_tensor.repeat(batchsize, 1, cfg.PGN_scan_X, cfg.PGN_scan_Y)
        data_pose_map = zero_tensor.repeat(batchsize, cfg.PGN_K_portion, scaled_X, scaled_Y)
        pmap = zero_tensor.repeat(1, 1, scaled_X, scaled_Y)

        if training:
            degree_each_portion = cfg.pi * 2 / cfg.PGN_K_portion

            for i in range(batchsize):
                index = iter * batchsize + i
                data_fixed_map[i], _ = self.fixed_map_train.__getitem__(self.training_index[index], (scaled_Y, scaled_X))
                data_scaninfo[i], _ = self.scan_info_train.__getitem__(self.training_index[index], (scaled_Y, scaled_X))
                pmap[0], _ = self.pmap_info_train.__getitem__(self.training_index[index], (scaled_Y, scaled_X))
                pmap_ = pmap[0] / torch.sum(pmap[0])

                rot = self.rotation_info_train.__getitem__(self.training_index[index])

                start_deg = torch.zeros(1).to(cfg.PGN_device)
                while rot > self.K_degree[int(start_deg)]:
                    start_deg += 1

                data_pose_map[i, int(start_deg), :, :] = (self.K_degree[int(start_deg)] - rot) / degree_each_portion
                data_pose_map[i, int(start_deg) - 1, :, :] = 1 - data_pose_map[i, int(start_deg), :, :]

                data_pose_map[i] = data_pose_map[i] * pmap_

            return data_fixed_map, data_scaninfo, data_pose_map

        else:

            degree_each_portion = cfg.pi * 2 / cfg.PGN_K_portion

            for i in range(batchsize):
                index = iter * batchsize + i
                data_fixed_map[i], _ = self.fixed_map_val.__getitem__(self.validation_index[index], (scaled_Y, scaled_X))
                data_scaninfo[i], _ = self.scan_info_val.__getitem__(self.validation_index[index], (scaled_Y, scaled_X))
                pmap[0], _ = self.pmap_info_val.__getitem__(self.validation_index[index], (scaled_Y, scaled_X))
                pmap_ = pmap[0] / torch.sum(pmap[0])

                rot = self.rotation_info_val.__getitem__(self.validation_index[index])

                start_deg = torch.zeros(1).to(cfg.PGN_device)
                while rot > self.K_degree[int(start_deg)]:
                    start_deg += 1

                data_pose_map[i, int(start_deg), :, :] = (self.K_degree[int(start_deg)] - rot) / degree_each_portion
                data_pose_map[i, int(start_deg) - 1, :, :] = 1 - data_pose_map[i, int(start_deg), :, :]

                data_pose_map[i] = data_pose_map[i] * pmap_

            return data_fixed_map, data_scaninfo, data_pose_map