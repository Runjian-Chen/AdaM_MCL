import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append("..")
from src.Particle_Filter import DL_Aided_Particle_Filter , MCL_Particle_Filter, Mixture_MCL_Particle_Filter, Adaptive_Particle_Filter_KD_tree, Mixture_MCL_Particle_Filter_DSOM
import cv2
import numpy as np
from src.utils import blue , green
import datetime
import os
import time
import math
from config.config import cfg

class Global_Localization_Interface():
    def __init__(self,dataset,subdataset,update_mode,particle_num,paras):
        self.dataset = dataset
        self.subdataset = subdataset
        self.particle_num = particle_num
        self.paras = paras
        self.update_mode = update_mode
        self.dataset_path = '../Dataset/' + dataset +'/Data_Transformed/Test_MCL/' + subdataset + '/'
        self.global_map = cv2.cvtColor(cv2.imread(self.dataset_path+'global_map.png'),cv2.COLOR_BGR2GRAY)

        self.observation = np.loadtxt(self.dataset_path+'laser.txt')
        self.gt_pose = np.loadtxt(self.dataset_path+'gt_pose.txt')
        self.odometry = np.loadtxt(self.dataset_path + 'odo.txt')
        odometry = np.zeros((self.odometry.shape[0]+1,self.odometry.shape[1]))
        odometry[1:,:]=self.odometry
        self.odometry = odometry

        self.particle_evolution = []
        self.error_in_position = []
        self.error_in_rotation = []
        self.time_consumption = []

        self.path4results = './logs4MCL/'+self.dataset+'/'+self.subdataset+'/'+update_mode+'_'+str(self.particle_num)+'_particles_'+str(paras['w_cut'])+'/'+'log_time_' + str(
            datetime.datetime.now()) + '_update_mode_' + self.update_mode + '_particle_num_' + str(self.particle_num) + '/'
        os.makedirs(self.path4results)

        self.paras['log_dir'] = self.path4results

        if update_mode == "Adaptive_Mixture_MCL":
            self.filter = DL_Aided_Particle_Filter(self.global_map,int(self.particle_num),self.paras)
        elif update_mode == "MCL":
            self.filter = MCL_Particle_Filter(self.global_map,int(self.particle_num),self.paras)
        elif update_mode == "Mixture_MCL":
            self.paras['sampling_data'] = np.load(self.dataset_path + 'dual_sampling_random.npy')
            self.filter = Mixture_MCL_Particle_Filter(self.global_map, int(self.particle_num), self.paras)

    def update_pf(self):
        for i in range(self.observation.shape[0]):
            print(blue('Update Iteration ' + str(i + 1)))
            print('Update Mode: ' + self.update_mode)
            print('Random_Produce: ' + str(self.paras['Random_Produce']))

            gt_x = self.gt_pose[i, 0]
            gt_y = self.gt_pose[i, 1]
            gt_theta = self.gt_pose[i, 2]
            gt_pmap = np.zeros(self.global_map.shape)
            gt_pmap[int(gt_x),int(gt_y)] = 255
            gt_pmap = cv2.GaussianBlur(gt_pmap, (7, 7), 3)
            cv2.normalize(gt_pmap, gt_pmap, 0, 255, cv2.NORM_MINMAX)
            x, y, theta, w = self.filter.Get_Particle_Set()
            particle_set = np.column_stack((x, y, theta, w))

            map_with_particles = self.global_map.copy()
            map_with_particles = np.expand_dims(map_with_particles, axis=-1)
            map_with_particles = np.repeat(map_with_particles,3,-1)

            for j in range(self.filter.N_particles):
                if w[j] != 0:
                    x_ = x[j]
                    y_ = y[j]
                    point = (int(x_), int(y_))
                    map_with_particles[point] = (0, int(255 * w[j] * 500), 0)

            map_with_particles = cv2.rectangle(map_with_particles, (int(gt_y - 1), int(gt_x - 1)),
                                               (int(gt_y + 1), int(gt_x + 1)), (0, 0, 255), -1)
            cv2.imwrite(self.path4results + str(i) + '_th_iter.png', map_with_particles)
            cv2.imwrite(self.path4results + str(self.filter.iteration) + '_gtpmap.png', gt_pmap)

            time_start = time.time()
            action_last = {'delta_x': self.odometry[i, 0], 'delta_y': self.odometry[i, 1],
                           'delta_theta': self.odometry[i, 2],
                           'var_odom_x': 0.01, 'var_odom_y': 0.01, 'var_odom_theta': 0.01}
            observation_now = self.observation[i]
            self.filter.update_pf(action_last, observation_now, paras=self.paras)
            time_end = time.time()

            est_x, est_y, est_theta, est_w = self.filter.Get_Estimation_Pose()

            self.particle_evolution.append(particle_set)
            self.error_in_position.append(math.sqrt((gt_x - est_x) ** 2 + (gt_y - est_y) ** 2))
            self.error_in_rotation.append(math.sqrt((gt_theta - est_theta) ** 2))
            self.time_consumption.append(time_end - time_start)
            self.global_step = self.filter.iteration

            print('Ground truth: ' + str(gt_x) + ' ' + str(gt_y) + ' ' + str(gt_theta))
            print('Estimation: ' + str(est_x) + ' ' + str(est_y) + ' ' + str(est_theta))
            print(green('Error in position: ') + str((gt_x - est_x) ** 2 + (gt_y - est_y) ** 2))
            print(green('Error in orientation: ') + str((gt_theta - est_theta) ** 2))
            print(green('Weight of estimation: ') + str(est_w))
            print(green('Time cost: ') + str(time_end - time_start))

        particle_set = np.array(self.particle_evolution)
        np.save(self.path4results + 'particle_set.npy', particle_set)
        np.savetxt(self.path4results + 'error_position.txt', np.array(self.error_in_position))
        np.savetxt(self.path4results + 'error_orientation.txt', np.array(self.error_in_rotation))
        np.savetxt(self.path4results + 'time_consumption.txt', np.array(self.time_consumption))