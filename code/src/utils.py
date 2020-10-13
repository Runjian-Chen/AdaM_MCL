import sys
sys.path.append("..")
from config.config import cfg
import math
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import os
green = lambda x: '\033[92m' + x + '\033[0m'
blue = lambda x: '\033[94m' + x + '\033[0m'

def check_collision(x, y, map_env):
    H, W = map_env.shape
    for i in range(-4, 4):
        for j in range(-4, 4):
            xx = x + i
            yy = y + j
            if int(xx) == 0 or int(xx) == H - 1 or int(yy) == 0 or int(yy) == W - 1 or map_env[
                xx, yy] > cfg.gray_threshold:
                return False

    return True


def ray_trace(x, y, map_env, degree, max_range, laser_scan_st_angle, steps, angle_per_step):
    H, W = map_env.shape
    scan = np.zeros((H, W))
    range_ = max_range ** 2

    for i in range(steps):

        x_axis_map = x
        y_axis_map = y
        radian = degree + laser_scan_st_angle + i * angle_per_step

        while int(x_axis_map) > 0 and int(x_axis_map) < H-1 and int(y_axis_map) > 0 and int(y_axis_map) < W-1 and (
                map_env[int(x_axis_map), int(y_axis_map)] <= cfg.gray_threshold):
            x_axis_base = y_axis_map
            y_axis_base = H - x_axis_map
            x_axis_base += math.cos(radian)
            y_axis_base += math.sin(radian)
            x_axis_map = H - y_axis_base
            y_axis_map = x_axis_base

        if int(x_axis_map) > 0 and int(x_axis_map) < H-1 and int(y_axis_map) > 0 and int(y_axis_map) < W-1 and (
                x - x_axis_map) ** 2 + (y - y_axis_map) ** 2 < range_:
            delta_x_axis_map = x_axis_map - x + np.random.normal(0, 0.01)
            delta_y_axis_map = y_axis_map - y + np.random.normal(0, 0.01)

            delta_x_axis_base = delta_y_axis_map
            delta_y_axis_base = -1 * delta_x_axis_map

            delta_x_axis_base_ = delta_x_axis_base * math.cos(-1 * degree) - delta_y_axis_base * math.sin(-1 * degree)
            delta_y_axis_base_ = delta_x_axis_base * math.sin(-1 * degree) + delta_y_axis_base * math.cos(-1 * degree)

            delta_x_axis_map = -1 * delta_y_axis_base_
            delta_y_axis_map = delta_x_axis_base_

            x_map = int(H // 2 + delta_x_axis_map)
            y_map = int(W // 2 + delta_y_axis_map)
            if x_map > 0 and x_map <  H-1 and y_map > 0 and y_map < W-1:
                scan[x_map,y_map] = 255

    return scan


class txt_loader(Dataset):
    def __init__(self, root_dir, sub_dir):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.filenames = os.listdir(self.root_dir + self.sub_dir)
        filenames = []
        for i in range(len(self.filenames)):
            if self.filenames[i].endswith(".txt") and (not self.filenames[i].startswith(".")):
                filenames.append(self.filenames[i])
        self.filenames = filenames
        self.len = len(self.filenames)
        self.filenames.sort(key=lambda x: int(x[:-4]))

    def __getitem__(self, index):

        filename = self.filenames[index]

        txt = np.loadtxt(self.root_dir + self.sub_dir + filename)

        gtpos = txt

        return torch.from_numpy(gtpos)

    def __len__(self):
        return self.len


class img_loader(Dataset):
    def __init__(self, root_dir, sub_dir):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.filenames = os.listdir(self.root_dir + self.sub_dir)
        filenames = []
        for i in range(len(self.filenames)):
            if self.filenames[i].endswith(".png") and (not self.filenames[i].startswith(".")):
                filenames.append(self.filenames[i])
        self.filenames = filenames
        self.len = len(self.filenames)
        self.filenames.sort(key=lambda x: int(x[:-4]))

    def __getitem__(self, index, scaled_size):

        filename = self.filenames[index]

        img = cv2.imread(self.root_dir + self.sub_dir + filename)

        img = img[:, :, 0]

        H, W = img.shape

        img = cv2.resize(img, scaled_size)

        im_info = torch.zeros(3)

        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0), im_info.unsqueeze(0)

    def __len__(self):
        return self.len


class img_loader_scan(Dataset):
    def __init__(self, root_dir, sub_dir):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.filenames = os.listdir(self.root_dir + self.sub_dir)
        filenames = []
        for i in range(len(self.filenames)):
            if self.filenames[i].endswith(".png") and (not self.filenames[i].startswith(".")):
                filenames.append(self.filenames[i])
        self.filenames = filenames
        self.len = len(self.filenames)
        self.filenames.sort(key=lambda x: int(x[:-4]))

    def __getitem__(self, index, scaled_size):

        filename = self.filenames[index]

        img = cv2.imread(self.root_dir + self.sub_dir + filename)

        img = img[:, :, 0]

        scaled_Y, scaled_X = scaled_size

        img = cv2.resize(img, (scaled_Y, scaled_X))

        img = img[scaled_X // 2 - cfg.DSOM_scan_X // 2:scaled_X // 2 + cfg.DSOM_scan_X // 2,
              scaled_Y // 2 - cfg.DSOM_scan_Y // 2:scaled_Y // 2 + cfg.DSOM_scan_Y // 2]

        im_info = torch.zeros(3)

        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0), im_info.unsqueeze(0)

    def __len__(self):
        return self.len
