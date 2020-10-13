import numpy as np
from easydict import EasyDict as edict
import yaml
import torch
import dxfgrabber

# Generate config
__C = edict()
cfg = __C

# Config for DSOM
__C.DSOM_model_path = ''
__C.DSOM_device = 'cpu'
__C.DSOM_K_portion = 32
__C.DSOM_X = 320
__C.DSOM_Y = 320
__C.DSOM_scan_X = 128
__C.DSOM_scan_Y = 128
__C.DSOM_scan_margin = 4
__C.DSOM_batchsize = 1
__C.DSOM_training_iters = 7200 * 4
__C.DSOM_validation_iters = 400
__C.DSOM_num_epoch = 100
__C.DSOM_zero_tensor = torch.zeros(1,1,1,1).to(__C.DSOM_device)

__C.Free = 0
__C.Occupied = 255
__C.green = lambda x: '\033[92m' + x + '\033[0m'
__C.blue = lambda x: '\033[94m' + x + '\033[0m'
__C.rad_min = -3.14159
__C.rad_max = 3.14159
__C.K_rad = np.linspace(-3.14159,3.14159,32)
__C.pi = 3.14159
__C.Dual_MCL_samples_path = ''

__C.Num_Sample_from_Previous_Set = 3
__C.Num_Sample_from_Observation = 10
__C.Weighting_Factor_for_Random_Samples = 0.8
__C.Weighting_Factor_for_Dual_MCL = 0.5

__C.global_map_path = './Test_Data/UPO/map.png'
__C.gt_path = ['./Test_Data/UPO/gt_pose_0.txt','./Test_Data/UPO/gt_pose_1.txt','./Test_Data/UPO/gt_pose_2.txt','./Test_Data/UPO/gt_pose_3.txt','./Test_Data/UPO/gt_pose_4.txt','./Test_Data/UPO/gt_pose_5.txt','./Test_Data/UPO/gt_pose_6.txt']
__C.observation_path = ['./Test_Data/UPO/laser_0.txt','./Test_Data/UPO/laser_1.txt','./Test_Data/UPO/laser_2.txt','./Test_Data/UPO/laser_3.txt','./Test_Data/UPO/laser_4.txt','./Test_Data/UPO/laser_5.txt','./Test_Data/UPO/laser_6.txt']
__C.odo_path = ['./Test_Data/UPO/odo_0.txt','./Test_Data/UPO/odo_1.txt','./Test_Data/UPO/odo_2.txt','./Test_Data/UPO/odo_3.txt','./Test_Data/UPO/odo_4.txt','./Test_Data/UPO/odo_5.txt','./Test_Data/UPO/odo_5.txt']
__C.steps_per_file = 50
__C.gray_threshold = 230

#################################
##
##
## Configures for the UPO dataset
##
##
#################################


# The transformation between poses and pixels on the map is described below
'''
    t, x, y, theta = odo[i]
    res_multiplier = 5.6
    x_drift = -12
    y_drift = 6
    env = cv2.imread(map_path)
    h , w , c = env.shape
    env[int(round(h-y*res_multiplier)+y_drift),int(round(x*res_multiplier)+x_drift),:]
'''

# Coordinate definition
'''
H, W = img.shape
matrix [H,W] standing for an images
[a_0_0 ... a_0_W-1
 a_1_0 ... a_1_W-1
 ...   ... .......
 a_H-1_0 ... a_H-1_W-1]

Map axis: Originate at [0,0] of the matrix
--------------> y_map
|
|
|
|
|
|
⬇️ x_map

Base axis: Originate at [H,0] of the matrix
⬆️ y_base
|
|
|
|
|
|
--------------> x_base

rotation: starting from the positive x_base axis and turning positive --> counterclockwise in the base axis
'''

# Transformation from map to scan, given pose and laser information
'''
    (x,y,degree) is the pose and map_env stands for the image

    x_axis_map = x
    y_axis_map = y
    radian = degree + cfg.UPO_Dataset_angle_min + i * cfg.UPO_Dataset_angle_increment
   
    while int(x_axis_map)>0 and int(x_axis_map)<H and int(y_axis_map)>0 and int(y_axis_map)<W and (map_env[int(x_axis_map),int(y_axis_map)] == 255):
      x_axis_base = y_axis_map
      y_axis_base = H - x_axis_map
      x_axis_base += math.cos(radian)
      y_axis_base += math.sin(radian)
      x_axis_map = H - y_axis_base
      y_axis_map = x_axis_base
    
    if int(x_axis_map)>0 and int(x_axis_map)<H and int(y_axis_map)>0 and int(y_axis_map)<W and (x-x_axis_map) ** 2 + (y-y_axis_map) ** 2 < range_:
      
      delta_x_axis_map = x_axis_map - x + np.random.normal(0, 0.1)
      delta_y_axis_map = y_axis_map - y + np.random.normal(0, 0.1)
      
      delta_x_axis_base = delta_y_axis_map
      delta_y_axis_base = -1 * delta_x_axis_map

      delta_x_axis_base_ = delta_x_axis_base*math.cos(-1*degree) - delta_y_axis_base*math.sin(-1*degree)
      delta_y_axis_base_ = delta_x_axis_base*math.sin(-1*degree) + delta_y_axis_base*math.cos(-1*degree)

      delta_x_axis_map = -1 * delta_y_axis_base_
      delta_y_axis_map = delta_x_axis_base_
      
      scan[int(H//2+delta_x_axis_map),int(W//2+delta_y_axis_map)] = 0
'''

# Transformation from ranges to the map is described below
'''
    theta_now = msg_bag.angle_min + i * msg_bag.angle_increment

    if not (math.isinf(msg_bag.ranges[i]) or math.isnan(msg_bag.ranges[i])):
        delta_x_axis_base = math.cos(theta_now) * msg_bag.ranges[i] * cfg.UPO_Dataset_res_multiplier
        delta_y_axis_base = math.sin(theta_now) * msg_bag.ranges[i] * cfg.UPO_Dataset_res_multiplier
        delta_x_axis_base_ = delta_x_axis_base*math.cos(theta) - delta_y_axis_base*math.sin(theta)
        delta_y_axis_base_ = delta_x_axis_base*math.sin(theta) + delta_y_axis_base*math.cos(theta)

        delta_x_axis_map = -1 * delta_y_axis_base_
        delta_y_axis_map = delta_x_axis_base_

        x_scan = int(x_map + delta_x_axis_map)
        y_scan = int(y_map + delta_y_axis_map)

        env[x_scan,y_scan,2] = 255
        env[x_scan,y_scan,1] = 0
        env[x_scan,y_scan,0] = 255
'''

# Occupancy definition
'''
  Free : 0
  Occupied : 255
'''

__C.UPO_Dataset_map_path = '../../Dataset/UPO/map.png'
__C.UPO_Dataset_root_path = '../../Dataset/UPO/'
__C.UPO_Dataset_selected_dataset = [1,2,3]
__C.UPO_Dataset_map_H = 880
__C.UPO_Dataset_map_W = 830
__C.UPO_Dataset_resolution = 1 / 5.6
__C.UPO_Dataset_x_drift = 6
__C.UPO_Dataset_y_drift = -12
__C.UPO_Dataset_scan_H = 880
__C.UPO_Dataset_scan_W = 830
__C.UPO_Dataset_angle_min = -1.57079637051
__C.UPO_Dataset_angle_increment = 0.00436332309619
__C.UPO_Dataset_angle_portions = 720
__C.UPO_Dataset_range_max = 60
__C.UPO_Augmented_Dataset_selected = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]


#################################
##
##
## Configures for the simulation dataset
##
##
#################################
__C.simulation_Dataset_root_path = '../../Dataset/simulation/'
__C.simulation_Dataset_selected_dataset = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
#_C.simulation_Dataset_selected_dataset = [1,2]
__C.simulation_Dataset_resolution = 1
__C.simulation_Dataset_max_range = 30
__C.simulation_Dataset_st_angle = - 179 / 180 * __C.pi
__C.simulation_Dataset_angle_step = 1 / 180 * __C.pi
__C.simulation_Dataset_scan_steps = 360

#################################
##
##
## Configures for the Bicocca dataset
##
##
#################################
__C.Bicocca_dxf_path = "../../Dataset/Bicocca/2009-02-25b/Drawings_02.dxf"
__C.Bicocca_gt_trajectory_path = "../../Dataset/Bicocca/2009-02-25b/Bicocca_2009-02-25b-GT-extended.csv"
__C.Bicocca_laser_scan_path = "../../Dataset/Bicocca/2009-02-25b/Bicocca_2009-02-25b-SICK_FRONT.csv"
__C.Bicocca_resolution = 0.2
__C.Bicocca_st_angle = - 90 / 180 * __C.pi
__C.Bicocca_angle_step = 1 / 180 * __C.pi
__C.Bicocca_scan_steps = 183
__C.Bicocca_crop = [(950,0,1400,400),(980,400,1400,750)]
__C.Bicocca_laser_max_range = 30 # here in meters
__C.Bicocca_dataset_root_path = '../../Dataset/Bicocca/'
__C.Bicocca_dataset = '2009-02-25b'