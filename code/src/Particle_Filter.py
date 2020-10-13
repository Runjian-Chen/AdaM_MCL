import numpy as np
import math
import scipy.stats
import torch
from config.config import cfg
from model.DSOM import DSOM
import sklearn.neighbors
import cv2
import sys
"""
    What these filters do:
        Input: environment map, odometry and observation at each time stamp. 
        Output: estimation of the robot pose at each time stamp in the map coordinate.
"""

"""
    Deep learning aided particle filter.
"""
class DL_Aided_Particle_Filter:
    # Initialization
    def __init__(self, global_map, N_particles, paras, Initial_pose = None, Initial_variance = None):
        ## global_map : [H,W] matrix for a occupancy map
        ## N_particles : Number of particles used in the algorithm
        ## paras : other parameters

        ## Initialize map of the environment
        self.global_map = global_map
        self.map_H , self.map_W = self.global_map.shape
        self.scale_factor = math.sqrt(cfg.DSOM_X*cfg.DSOM_Y/(self.map_H*self.map_W))
        self.scaled_X = int(self.map_H * self.scale_factor)
        self.scaled_Y = int(self.map_W * self.scale_factor)
        self.laser_steps = paras['laser_steps']

        ## Initialize iteration
        self.iteration = 0

        ## Initialize particle number
        self.N_particles = N_particles

        ## Initialize pose generation network
        self.DSOM = DSOM(1, 1)
        self.DSOM.to(cfg.DSOM_device)
        check_point = torch.load(cfg.DSOM_model_path, cfg.DSOM_device)
        self.DSOM.load_state_dict(check_point)
        self.DSOM.eval()
        self.K_degree = np.linspace(cfg.rad_min, cfg.rad_max, cfg.DSOM_K_portion)

        self.KDtree_particles = None
        self.x = np.zeros(self.N_particles)
        self.y = np.zeros(self.N_particles)
        self.theta = np.zeros(self.N_particles)
        self.w = np.zeros(self.N_particles)
        self.w_normalized = np.zeros(self.N_particles)

        self.Free_rep = cfg.Free
        free_space = np.where(global_map<cfg.gray_threshold)
        self.free_map = np.ones((self.map_H , self.map_W)) * abs(255-self.Free_rep)
        self.free_map[free_space] = self.Free_rep
        self.free_map = abs(255 - self.free_map)
        self.free_map = torch.from_numpy(self.free_map).unsqueeze(0).to(cfg.DSOM_device)
        self.map_binary_trans = global_map.copy()
        index = np.where(self.map_binary_trans>cfg.gray_threshold)
        index_ = np.where(self.map_binary_trans<=cfg.gray_threshold)
        self.map_binary_trans[index] = 0
        self.map_binary_trans[index_] = 255
        
        self.pmap = np.zeros((cfg.DSOM_K_portion,cfg.DSOM_X,cfg.DSOM_Y))
        self.est_w = 0
        self.w_perfect = (paras['laser_portions'] / self.laser_steps) * (( scipy.stats.norm.pdf(0,0,paras['var_rc_weighting_model'])))

        # Init particles
        if Initial_pose == None and Initial_variance == None:
            for i in range(self.N_particles):
                x = np.random.uniform(low=0, high=self.map_H)
                y = np.random.uniform(low=0, high=self.map_W)
                while not (self.global_map[int(x),int(y)] == self.Free_rep):
                    x = np.random.uniform(low=0, high=self.map_H)
                    y = np.random.uniform(low=0, high=self.map_W)
                self.x[i] = x
                self.y[i] = y
                self.theta[i] = np.random.uniform(low=cfg.rad_min, high=cfg.rad_max)

            self.w = np.ones(N_particles) * 1.0 / N_particles
            self.w_normalized = self.w
            self.est_x = np.sum(self.x * self.w_normalized,keepdims=True)
            self.est_y = np.sum(self.y * self.w_normalized,keepdims=True)
            self.est_theta = np.sum(self.theta * self.w_normalized,keepdims=True)
        
    # Function for updating particle set
    def update_pf(self, action, observation, paras = None):
        ## action: odometry described as delta_x, delta_y, delta_theta
        ## observation: 2D scan denoted as a vector
        ## paras: parameters for this mode

        # Add up iterations
        self.iteration += 1

        # Convert the scan vector to image [128,128]
        scan_on_image = self.scan_v2img(observation,paras)
        cv2.imwrite(paras['log_dir']+str(self.iteration)+'_scan.png',scan_on_image)

        # Load the scan and global map to device we use
        scan = cv2.resize(scan_on_image,(self.scaled_Y,self.scaled_X))
        scan = scan[self.scaled_X//2-cfg.DSOM_scan_X//2:self.scaled_X//2+cfg.DSOM_scan_X//2,self.scaled_Y//2-cfg.DSOM_scan_Y//2:self.scaled_Y//2+cfg.DSOM_scan_Y//2]
        scan = torch.from_numpy(scan).squeeze().unsqueeze(0).unsqueeze(0).to(cfg.DSOM_device).float()
        g_map = torch.from_numpy(cv2.resize(self.global_map,(self.scaled_Y,self.scaled_X))).squeeze().unsqueeze(0).unsqueeze(0).to(cfg.DSOM_device).float()

        # Generate probability map : [32,128,128]
        pmap = torch.exp(self.DSOM(scan,g_map,cfg.DSOM_zero_tensor)).view(1,cfg.DSOM_K_portion,self.scaled_X,self.scaled_Y)

        # Broadcast the pmap to original size
        upsample = torch.nn.Upsample((self.map_H , self.map_W),mode='bilinear')
        pmap = upsample(pmap).squeeze()

        pmap = pmap * self.free_map

        # Normalize pmap
        pmap = pmap / torch.sum(pmap)

        # Convert pmap to an image to show the prediction result
        pmap_ = torch.sum(pmap,dim=0).detach().cpu().numpy()
        cv2.normalize(pmap_,pmap_,0,255,cv2.NORM_MINMAX)
        cv2.imwrite(paras['log_dir']+str(self.iteration)+'_pmap.png',pmap_)

        pmap = pmap.detach().cpu().numpy()
        self.pmap = pmap
        pmap = pmap.reshape(-1)
        pmap = pmap / np.sum(pmap)

        # New particle set
        x_new = np.zeros(self.N_particles)
        y_new = np.zeros(self.N_particles)
        theta_new = np.zeros(self.N_particles)
        w_new = np.zeros(self.N_particles)
        
        print('DL Aided MCL updating......')
            
        print('Building KDtree for previous time step......')
        # Building KDtree for particles at previous time step
        previous_set = np.zeros((self.N_particles,3))

        previous_set[:,0] = self.x.copy() + np.random.normal(action['delta_x'], action['var_odom_x'],self.N_particles)
        previous_set[:,1] = self.y.copy() + np.random.normal(action['delta_y'], action['var_odom_y'],self.N_particles)
        previous_set[:,2] = self.theta.copy() + np.random.normal(action['delta_theta'], action['var_odom_theta'],self.N_particles)
        self.KDtree_particles = sklearn.neighbors.KDTree(previous_set)
        print('Finish building KDtree.')

        print('Updating particle set......')

        # Divide the whole particles set into two parts
        count_mcl = 0
        index_motion = []
        for i in range (self.N_particles):
            
            if self.iteration == 1 :
                p_MCL = 1
            else:
                p_MCL = self.w[i] / self.w_perfect
            
            if p_MCL > paras['w_cut'] : 
                p_MCL = 1

            # randomly select update mode
            p_random = np.random.uniform(low=0, high=1)

            if p_random < p_MCL:
                ## Motion model update
                index_motion.append(i)
                count_mcl += 1


        # MCL update
        if count_mcl > 0:
            ## Get the set for MCL update branch
            index_motion = np.array(index_motion)
            x_motion = self.x[index_motion].copy()
            y_motion = self.y[index_motion].copy()
            theta_motion = self.theta[index_motion].copy()
            w_motion = self.w_normalized[index_motion].copy()
            w_motion = w_motion / np.sum(w_motion)
            ## Sample particles based on previous weights
            index_sampled = np.random.choice(count_mcl, count_mcl, p=w_motion)
            x_new[:count_mcl] = x_motion[index_sampled]
            y_new[:count_mcl] = y_motion[index_sampled]
            theta_new[:count_mcl] = theta_motion[index_sampled]
            ## Update newly sampled set based on motion model
            x_new[:count_mcl] += np.random.normal(action['delta_x'], action['var_odom_x'],count_mcl)
            y_new[:count_mcl] += np.random.normal(action['delta_y'], action['var_odom_y'],count_mcl)
            theta_new[:count_mcl] += np.random.normal(action['delta_theta'], action['var_odom_theta'],count_mcl)
            ## Weight these particles with ray casting algorithm
            w_new[:count_mcl] = self.Ray_Casting_Weighting(x_new[:count_mcl],y_new[:count_mcl],theta_new[:count_mcl],self.global_map,observation,paras)

        # Dual MCL update
        if count_mcl < self.N_particles:
        
            if np.sum(pmap) > 0 :
                ## Sample particles from pmap
                index = np.random.choice(len(pmap), self.N_particles-count_mcl, p=pmap)
                theta_new[count_mcl:] = cfg.K_rad[(index // (self.map_H*self.map_W)).astype(int)]
                index = index % (self.map_H*self.map_W)
                x_new[count_mcl:] = index // self.map_W
                index = index % self.map_W
                y_new[count_mcl:] = index

            else : 
                for i in range(self.N_particles-count_mcl):
                    x = np.random.uniform(low=0, high=self.map_H)
                    y = np.random.uniform(low=0, high=self.map_W)
                    while not (self.global_map[int(x),int(y)] == self.Free_rep):
                        x = np.random.uniform(low=0, high=self.map_H)
                        y = np.random.uniform(low=0, high=self.map_W)
                    x_new[count_mcl+i] = x
                    y_new[count_mcl+i] = y
                    theta_new[count_mcl+i] = np.random.uniform(low=cfg.rad_min, high=cfg.rad_max)
                
            # Sample particle at former time frame
            index = self.find_index_from_previous_set(x_new[count_mcl:],y_new[count_mcl:],theta_new[count_mcl:], action)
            index = index.astype(int)
            # Weight new particle
            w_new[count_mcl:] = self.w[index].copy()

        print('Finish updating particle set.')
        print('Update statistics: ')
        print('Motion update: '+str(count_mcl))
        print('Sampling update: '+str(self.N_particles-count_mcl))

        # Update particle set
        self.x = x_new
        self.y = y_new

        # Convert orientation into [cfg.rad_min,cfg.rad_max]
        for i in range(self.N_particles):
            while theta_new[i] < cfg.rad_min :
                theta_new[i] += 2 * cfg.pi
            while theta_new[i] > cfg.rad_max :
                theta_new[i] -= 2 * cfg.pi

        self.theta = theta_new
        self.w = w_new
        self.w_normalized = np.true_divide(w_new,np.sum(w_new))

        self.est_x = np.sum(self.x * self.w_normalized,keepdims=True)
        self.est_y = np.sum(self.y * self.w_normalized,keepdims=True)
        self.est_theta = np.sum(self.theta * self.w_normalized,keepdims=True)
        self.est_w = np.sum(self.Ray_Casting_Weighting(self.est_x,self.est_y,self.est_theta,self.global_map,observation,paras),keepdims=True)

    def Ray_Casting_Weighting(self, x, y, theta, global_map, observation, paras):

        scan_res = paras['scan_res']
        scan_angle_st = paras['scan_angle_st']
        res = paras['resolution']
        max_scan_range = paras['max_scan_range']
        var_rc_weighting_model = paras['var_rc_weighting_model']
        portions = paras['laser_portions']
        w = np.zeros(len(x))
        
        for index in range(len(x)):

            if np.isnan(x[index]) or np.isnan(y[index]) or x[index] >= self.map_H or x[index] <= 0 or y[index] >= self.map_W or y[index] <= 0 :#or  not (global_map[int(x[index]),int(y[index])] > cfg.gray_threshold):
                w[index] = 0
            else:
                for p in range(0,portions,self.laser_steps):
                    x_now = x[index].copy()
                    y_now = y[index].copy()
                    theta_now = theta[index].copy() + p * scan_res + scan_angle_st

                    while x_now < self.map_H and x_now > 0 and y_now < self.map_W and y_now > 0 and global_map[int(x_now),int(y_now)] < cfg.gray_threshold:
                        x_axis_base = y_now
                        y_axis_base = self.map_H - x_now
                        x_axis_base += math.cos(theta_now)
                        y_axis_base += math.sin(theta_now)
                        x_now = self.map_H - y_axis_base
                        y_now = x_axis_base

                    rt_len = math.sqrt((x_now-x[index]) ** 2 + (y_now-y[index]) ** 2)

                    if x_now < self.map_H and x_now > 0 and y_now < self.map_W and y_now > 0 and rt_len < max_scan_range/res:
                        rt_v = rt_len
                    else :
                        rt_v = float('inf')

                    if math.isinf(rt_v) and (math.isinf(observation[p]) or math.isnan(observation[p])):
                        p_hit = scipy.stats.norm.pdf(0,0,var_rc_weighting_model)
                    elif math.isnan(observation[p]) or math.isinf(observation[p]):
                        p_hit = 0
                    else:
                        diff = rt_v - (observation[p]/paras['resolution'])
                        p_hit = scipy.stats.norm.pdf(diff,0,var_rc_weighting_model)

                    w[index] += p_hit


        return w

    def find_index_from_previous_set(self, x_new, y_new, theta_new, action):

        len_new_set = len(x_new)
        index = np.zeros(len_new_set)

        current_set = np.zeros((len_new_set,3))
        current_set[:,0] = x_new
        current_set[:,1] = y_new
        current_set[:,2] = theta_new

        dist , ind = self.KDtree_particles.query(current_set, cfg.Num_Sample_from_Previous_Set)

        dist = 1 / (dist + 0.0000000001)

        for i in range(len_new_set):
            p = dist[i,:] / np.sum(dist[i,:])
            index_ = np.random.choice(cfg.Num_Sample_from_Previous_Set,1,p=p)
            index[i] = ind[i,index_]

        return index

    def rad2dis(self,theta):
        K = 0
        while K < cfg.DSOM_K_portion and self.K_degree[K]<theta:
            K+=1

        return K

    def scan_v2img(self,observation,paras):

        scan_map = np.zeros((self.map_H , self.map_W))
        
        center_x = self.map_H  // 2
        center_y = self.map_W // 2

        scan_res = paras['scan_res']
        scan_angle_st = paras['scan_angle_st']
        portions = paras['laser_portions']

        for p in range(portions):

            if (not (math.isinf(observation[p]) or math.isnan(observation[p]) or observation[p] > paras['max_scan_range'])):

                theta_now = p * scan_res + scan_angle_st

                delta_x_axis_base = math.cos(theta_now) * observation[p] / paras['resolution']
                delta_y_axis_base = math.sin(theta_now) * observation[p] / paras['resolution']
                
                delta_x_axis_map = -1 * delta_y_axis_base
                delta_y_axis_map = delta_x_axis_base
                
                x_scan = int(center_x + delta_x_axis_map)
                y_scan = int(center_y + delta_y_axis_map)
                if x_scan < self.map_H and y_scan < self.map_W:
                    scan_map[x_scan,y_scan] = 255

        return scan_map

    def Get_Particle_Set(self):

        return self.x, self.y, self.theta, self.w

    def Get_Estimation_Pose(self):

        return self.est_x, self.est_y, self.est_theta, self.est_w

class MCL_Particle_Filter:
    # Initialization
    def __init__(self, global_map, N_particles, paras, Initial_pose=None, Initial_variance=None):
        ## global_map : [H,W] matrix for a occupancy map
        ## N_particles : Number of particles used in the algorithm
        ## paras : other parameters

        ## Initialize map of the environment
        self.global_map = global_map
        self.map_H, self.map_W = self.global_map.shape
        self.laser_steps = paras['laser_steps']

        ## Initialize iteration
        self.iteration = 0

        ## Initialize particle number
        self.N_particles = N_particles

        ## Initialize pose generation network
        self.x = np.zeros(self.N_particles)
        self.y = np.zeros(self.N_particles)
        self.theta = np.zeros(self.N_particles)
        self.w_normalized = np.zeros(self.N_particles)
        self.w_perfect = (paras['laser_portions'] / self.laser_steps) * (
        (scipy.stats.norm.pdf(0, 0, paras['var_rc_weighting_model']) * paras['rc_w_hit']))

        if paras['Random_Produce'] == True:
            self.random_sample = int(float(paras['random_sample_rate'])* self.N_particles)
            self.mcl_sample = self.N_particles - self.random_sample
        else:
            self.random_sample = 0
            self.mcl_sample = self.N_particles - self.random_sample

        # Init particles
        if Initial_pose == None and Initial_variance == None:
            for i in range(self.N_particles):
                x = np.random.uniform(low=0, high=self.map_H)
                y = np.random.uniform(low=0, high=self.map_W)
                while not (self.global_map[int(x), int(y)] == cfg.Free):
                    x = np.random.uniform(low=0, high=self.map_H)
                    y = np.random.uniform(low=0, high=self.map_W)
                self.x[i] = x
                self.y[i] = y
                self.theta[i] = np.random.uniform(low=cfg.rad_min, high=cfg.rad_max)

            self.w_normalized = np.ones(N_particles) * 1.0 / N_particles
            self.est_x = np.sum(self.x * self.w_normalized, keepdims=True)
            self.est_y = np.sum(self.y * self.w_normalized, keepdims=True)
            self.est_theta = np.sum(self.theta * self.w_normalized, keepdims=True)

    # Function for updating particle set
    def update_pf(self, action, observation, paras=None):
        ## action: odometry described as delta_x, delta_y, delta_theta
        ## observation: 2D scan denoted as a vector
        ## paras: parameters for this mode

        # Add up iterations
        self.iteration += 1

        # New particle set
        x_new = np.zeros(self.N_particles)
        y_new = np.zeros(self.N_particles)
        theta_new = np.zeros(self.N_particles)

        # Sample particles at former time frame by their weights
        index_sampled = np.random.choice(self.N_particles, self.mcl_sample, p=self.w_normalized)
        x_new[:self.mcl_sample] = self.x[index_sampled]
        y_new[:self.mcl_sample] = self.y[index_sampled]
        theta_new[:self.mcl_sample] = self.theta[index_sampled]

        # Renew particles based on old particle and action control
        x_new[:self.mcl_sample] += np.random.normal(action['delta_x'], action['var_odom_x'], self.mcl_sample)
        y_new[:self.mcl_sample] += np.random.normal(action['delta_y'], action['var_odom_y'], self.mcl_sample)
        theta_new[:self.mcl_sample] += np.random.normal(action['delta_theta'], action['var_odom_theta'], self.mcl_sample)

        for i in range(self.random_sample):
            x = np.random.uniform(low=0, high=self.map_H)
            y = np.random.uniform(low=0, high=self.map_W)
            while self.global_map[int(x), int(y)] == cfg.Occupied:
                x = np.random.uniform(low=0, high=self.map_H)
                y = np.random.uniform(low=0, high=self.map_W)
            x_new[self.mcl_sample + i] = x
            y_new[self.mcl_sample + i] = y
            theta_new[self.mcl_sample + i] = np.random.uniform(low=cfg.rad_min, high=cfg.rad_max)

        for i in range(self.N_particles):
            while theta_new[i] < cfg.rad_min:
                theta_new[i] += 2 * cfg.pi
            while theta_new[i] > cfg.rad_max:
                theta_new[i] -= 2 * cfg.pi

        # Weight new particle
        w_new = self.Ray_Casting_Weighting(x_new, y_new, theta_new, self.global_map, observation, paras)

        # Update particle set
        self.x = x_new
        self.y = y_new
        self.theta = theta_new
        self.w_normalized = w_new / np.sum(w_new)

        self.est_x = np.sum(self.x * self.w_normalized, keepdims=True)
        self.est_y = np.sum(self.y * self.w_normalized, keepdims=True)
        self.est_theta = np.sum(self.theta * self.w_normalized, keepdims=True)
        self.est_w = np.sum(
            self.Ray_Casting_Weighting(self.est_x, self.est_y, self.est_theta, self.global_map, observation, paras),
            keepdims=True)

    def Ray_Casting_Weighting(self, x, y, theta, global_map, observation, paras):

        scan_res = paras['scan_res']
        scan_angle_st = paras['scan_angle_st']
        max_scan_range = paras['max_scan_range']
        res = paras['resolution']
        var_rc_weighting_model = paras['var_rc_weighting_model']
        portions = paras['laser_portions']
        w = np.zeros(len(x))

        for index in range(len(x)):

            if np.isnan(x[index]) or np.isnan(y[index]) or x[index] >= self.map_H or x[index] <= 0 or y[
                index] >= self.map_W or y[
                index] <= 0:  # or  not (global_map[int(x[index]),int(y[index])] > cfg.gray_threshold):
                w[index] = 0
            else:
                for p in range(0, portions, self.laser_steps):
                    x_now = x[index].copy()
                    y_now = y[index].copy()
                    theta_now = theta[index].copy() + p * scan_res + scan_angle_st

                    while x_now < self.map_H and x_now > 0 and y_now < self.map_W and y_now > 0 and global_map[
                        int(x_now), int(y_now)] < cfg.gray_threshold:
                        x_axis_base = y_now
                        y_axis_base = self.map_H - x_now
                        x_axis_base += math.cos(theta_now)
                        y_axis_base += math.sin(theta_now)
                        x_now = self.map_H - y_axis_base
                        y_now = x_axis_base

                    rt_len = math.sqrt((x_now - x[index]) ** 2 + (y_now - y[index]) ** 2)

                    if x_now < self.map_H and x_now > 0 and y_now < self.map_W and y_now > 0 and rt_len < max_scan_range / res:
                        rt_v = rt_len
                    else:
                        rt_v = float('inf')

                    if math.isinf(rt_v) and (math.isinf(observation[p]) or math.isnan(observation[p])):
                        p_hit = scipy.stats.norm.pdf(0, 0, var_rc_weighting_model)
                    elif math.isnan(observation[p]) or math.isinf(observation[p]):
                        p_hit = 0
                    else:
                        #diff = rt_v - (observation[p] / cfg.Bicocca_resolution)
                        diff = rt_v - (observation[p])
                        p_hit = scipy.stats.norm.pdf(diff, 0, var_rc_weighting_model)

                    # Good on some dataset
                    #p = paras['rc_w_hit'] * p_hit + paras['rc_w_rand'] * 1.0 / max_scan_range
                    #w[index] += p ** 3
                    p = paras['rc_w_hit'] * p_hit + paras['rc_w_rand'] * 1.0 / (max_scan_range/res)
                    w[index] += p ** 3

        return w

    def find_index_from_previous_set(self, x_new, y_new, theta_new, action):

        len_new_set = len(x_new)
        index = np.zeros(len_new_set)

        current_set = np.zeros((len_new_set, 3))
        current_set[:, 0] = x_new
        current_set[:, 1] = y_new
        current_set[:, 2] = theta_new

        dist, ind = self.KDtree_particles.query(current_set, cfg.Num_Sample_from_Previous_Set)

        dist = 1 / (dist + 0.0000000001)

        for i in range(len_new_set):
            p = dist[i, :] / np.sum(dist[i, :])
            index_ = np.random.choice(cfg.Num_Sample_from_Previous_Set, 1, p=p)
            index[i] = ind[i, index_]

        return index

    def rad2dis(self, theta):
        K = 0
        while K < cfg.DSOM_K_portion and self.K_degree[K] < theta:
            K += 1

        return K


    def Get_Particle_Set(self):

        return self.x, self.y, self.theta, self.w_normalized

    def Get_Estimation_Pose(self):

        return self.est_x, self.est_y, self.est_theta, self.est_w

class Mixture_MCL_Particle_Filter:
    # Initialization
    def __init__(self, global_map, N_particles, paras, Initial_pose=None, Initial_variance=None):
        ## global_map : [H,W] matrix for a occupancy map
        ## N_particles : Number of particles used in the algorithm
        ## paras : other parameters

        ## Initialize map of the environment
        self.global_map = global_map
        self.map_H, self.map_W = self.global_map.shape
        self.laser_steps = paras['laser_steps']

        ## Initialize iteration
        self.iteration = 0

        ## Initialize particle number
        self.N_particles = N_particles

        ## Initialize pose generation network
        self.x = np.zeros(self.N_particles)
        self.y = np.zeros(self.N_particles)
        self.theta = np.zeros(self.N_particles)
        self.w_normalized = np.zeros(self.N_particles)
        self.w_perfect = (paras['laser_portions'] / self.laser_steps) * (
        (scipy.stats.norm.pdf(0, 0, paras['var_rc_weighting_model']) * paras['rc_w_hit']))

        # Init particles
        if Initial_pose == None and Initial_variance == None:
            for i in range(self.N_particles):
                x = np.random.uniform(low=0, high=self.map_H)
                y = np.random.uniform(low=0, high=self.map_W)
                while not (self.global_map[int(x), int(y)] == cfg.Free):
                    x = np.random.uniform(low=0, high=self.map_H)
                    y = np.random.uniform(low=0, high=self.map_W)
                self.x[i] = x
                self.y[i] = y
                self.theta[i] = np.random.uniform(low=cfg.rad_min, high=cfg.rad_max)

            self.w_normalized = np.ones(N_particles) * 1.0 / N_particles
            self.est_x = np.sum(self.x * self.w_normalized, keepdims=True)
            self.est_y = np.sum(self.y * self.w_normalized, keepdims=True)
            self.est_theta = np.sum(self.theta * self.w_normalized, keepdims=True)

        # Initialize sampling tree
        # sampling_data:[N,6] where N is the number of samples and 6 features includes (x,y,theta) and (f_1,f_2,f_3).
        self.sampling_data = paras['sampling_data']

        print('Building KDtree for sampling......')
        # Building KDtree for particles at previous time step
        self.KDtree_sampling = sklearn.neighbors.KDTree(self.sampling_data[:,3:6])
        self.sample_num = 5
        print('Finish building KDtree for sampling.')

    # Function for updating particle set
    def update_pf(self, action, observation, paras=None):
        ## action: odometry described as delta_x, delta_y, delta_theta
        ## observation: 2D scan denoted as a vector
        ## paras: parameters for this mode

        # Add up iterations
        self.iteration += 1

        # randomly select update mode
        p_random = np.random.uniform(low=0, high=1)

        if p_random < paras['P_MCL']:
            # Traditional MCL updating

            # New particle set
            x_new = np.zeros(self.N_particles)
            y_new = np.zeros(self.N_particles)
            theta_new = np.zeros(self.N_particles)

            # Sample particles at former time frame by their weights
            index_sampled = np.random.choice(self.N_particles, self.N_particles, p=self.w_normalized)
            x_new[:] = self.x[index_sampled]
            y_new[:] = self.y[index_sampled]
            theta_new[:] = self.theta[index_sampled]

            # Renew particles based on old particle and action control
            x_new[:] += np.random.normal(action['delta_x'], action['var_odom_x'], self.N_particles)
            y_new[:] += np.random.normal(action['delta_y'], action['var_odom_y'], self.N_particles)
            theta_new[:] += np.random.normal(action['delta_theta'], action['var_odom_theta'], self.N_particles)

            for i in range(self.N_particles):
                while theta_new[i] < cfg.rad_min:
                    theta_new[i] += 2 * cfg.pi
                while theta_new[i] > cfg.rad_max:
                    theta_new[i] -= 2 * cfg.pi

            # Weight new particle
            w_new = self.Ray_Casting_Weighting(x_new, y_new, theta_new, self.global_map, observation, paras)

        else:
            # Dual MCL updating

            # New particle set
            observation_feature = self.ob2feat(observation).reshape((1,-1))
            observation_feature = np.repeat(observation_feature, self.N_particles, axis=0)

            dist, ind = self.KDtree_sampling.query(observation_feature, self.sample_num)

            #dist = dist[:,1:]
            #ind = ind[:,1:]

            index = np.zeros(self.N_particles)

            dist = 1 / (dist + 0.0000000001)

            for i in range(self.N_particles):
                p = dist[i, :] / np.sum(dist[i, :])
                index_ = np.random.choice(self.sample_num, 1, p=p)
                index[i] = ind[i, index_]

            index = index.astype(int)
            x_new = self.sampling_data[index,0]
            y_new = self.sampling_data[index,1]
            theta_new = self.sampling_data[index,2]

            # Sample particle at former time frame
            index = self.find_index_from_previous_set(x_new, y_new, theta_new,
                                                      action)
            index = index.astype(int)
            # Weight new particle
            w_new = self.w_normalized[index].copy()

        # Update particle set
        if np.sum(w_new) == 0:
            for i in range(self.N_particles):
                x = np.random.uniform(low=0, high=self.map_H)
                y = np.random.uniform(low=0, high=self.map_W)
                while not (self.global_map[int(x), int(y)] == cfg.Free):
                    x = np.random.uniform(low=0, high=self.map_H)
                    y = np.random.uniform(low=0, high=self.map_W)
                self.x[i] = x
                self.y[i] = y
                self.theta[i] = np.random.uniform(low=cfg.rad_min, high=cfg.rad_max)
            self.w_normalized = np.ones(self.N_particles) * 1.0 / self.N_particles
        else:
            self.x = x_new
            self.y = y_new
            self.theta = theta_new
            self.w_normalized = w_new / np.sum(w_new)

        self.est_x = np.sum(self.x * self.w_normalized, keepdims=True)
        self.est_y = np.sum(self.y * self.w_normalized, keepdims=True)
        self.est_theta = np.sum(self.theta * self.w_normalized, keepdims=True)
        self.est_w = np.sum(
            self.Ray_Casting_Weighting(self.est_x, self.est_y, self.est_theta, self.global_map, observation, paras),
            keepdims=True)

    def Ray_Casting_Weighting(self, x, y, theta, global_map, observation, paras):

        scan_res = paras['scan_res']
        scan_angle_st = paras['scan_angle_st']
        max_scan_range = paras['max_scan_range']
        res = paras['resolution']
        var_rc_weighting_model = paras['var_rc_weighting_model']
        portions = paras['laser_portions']
        w = np.zeros(len(x))

        for index in range(len(x)):

            if np.isnan(x[index]) or np.isnan(y[index]) or x[index] >= self.map_H or x[index] <= 0 or y[
                index] >= self.map_W or y[
                index] <= 0 or   (global_map[int(x[index]),int(y[index])] > cfg.gray_threshold):
                w[index] = 0
            else:
                for p in range(0, portions, self.laser_steps):
                    x_now = x[index].copy()
                    y_now = y[index].copy()
                    theta_now = theta[index].copy() + p * scan_res + scan_angle_st

                    while x_now < self.map_H and x_now > 0 and y_now < self.map_W and y_now > 0 and global_map[
                        int(x_now), int(y_now)] < cfg.gray_threshold:
                        x_axis_base = y_now
                        y_axis_base = self.map_H - x_now
                        x_axis_base += math.cos(theta_now)
                        y_axis_base += math.sin(theta_now)
                        x_now = self.map_H - y_axis_base
                        y_now = x_axis_base

                    rt_len = math.sqrt((x_now - x[index]) ** 2 + (y_now - y[index]) ** 2)

                    if x_now < self.map_H and x_now > 0 and y_now < self.map_W and y_now > 0 and rt_len < max_scan_range/res:
                        rt_v = rt_len
                    else:
                        rt_v = float('inf')

                    if math.isinf(rt_v) and (math.isinf(observation[p]) or math.isnan(observation[p])):
                        p_hit = scipy.stats.norm.pdf(0, 0, var_rc_weighting_model)
                    elif math.isnan(observation[p]) or math.isinf(observation[p]):
                        p_hit = 0
                    else:
                        diff = rt_v - (observation[p] / paras['resolution'])
                        p_hit = scipy.stats.norm.pdf(diff, 0, var_rc_weighting_model)

                    p = paras['rc_w_hit'] * p_hit + paras['rc_w_rand'] * 1.0 / (max_scan_range / res)
                    w[index] += p ** 3

        return w

    def find_index_from_previous_set(self, x_new, y_new, theta_new, action):

        print('Building KDtree for previous time step......')
        # Building KDtree for particles at previous time step
        previous_set = np.zeros((self.N_particles, 3))

        previous_set[:, 0] = self.x.copy() + np.random.normal(action['delta_x'], action['var_odom_x'], self.N_particles)
        previous_set[:, 1] = self.y.copy() + np.random.normal(action['delta_y'], action['var_odom_y'], self.N_particles)
        previous_set[:, 2] = self.theta.copy() + np.random.normal(action['delta_theta'], action['var_odom_theta'],
                                                                  self.N_particles)
        KDtree = sklearn.neighbors.KDTree(previous_set)
        print('Finish building KDtree.')

        len_new_set = len(x_new)
        index = np.zeros(len_new_set)

        current_set = np.zeros((len_new_set, 3))
        current_set[:, 0] = x_new
        current_set[:, 1] = y_new
        current_set[:, 2] = theta_new

        dist, ind = KDtree.query(current_set, cfg.Num_Sample_from_Previous_Set)

        dist = 1 / (dist + 0.0000000001)

        for i in range(len_new_set):
            p = dist[i, :] / np.sum(dist[i, :])
            index_ = np.random.choice(cfg.Num_Sample_from_Previous_Set, 1, p=p)
            index[i] = ind[i, index_]

        return index

    def rad2dis(self, theta):
        K = 0
        while K < cfg.DSOM_K_portion and self.K_degree[K] < theta:
            K += 1

        return K

    def scan_v2img(self, observation, paras):

        scan_map = np.zeros((self.map_H, self.map_W))

        center_x = self.map_H // 2
        center_y = self.map_W // 2

        scan_res = paras['scan_res']
        scan_angle_st = paras['scan_angle_st']
        portions = paras['laser_portions']

        for p in range(portions):

            if (not (math.isinf(observation[p]) or math.isnan(observation[p]) or observation[p] > 30)):

                theta_now = p * scan_res + scan_angle_st

                delta_x_axis_base = math.cos(theta_now) * observation[p] / paras['resolution']
                delta_y_axis_base = math.sin(theta_now) * observation[p] / paras['resolution']

                delta_x_axis_map = -1 * delta_y_axis_base
                delta_y_axis_map = delta_x_axis_base

                x_scan = int(center_x + delta_x_axis_map)
                y_scan = int(center_y + delta_y_axis_map)
                if x_scan < self.map_H and y_scan < self.map_W:
                    scan_map[x_scan, y_scan] = 255

        return scan_map

    def Get_Particle_Set(self):

        return self.x, self.y, self.theta, self.w_normalized

    def Get_Estimation_Pose(self):

        return self.est_x, self.est_y, self.est_theta, self.est_w

    def ob2feat(self,observation):
        feat = np.zeros(3)
        cx = 0
        cy = 0
        A = 0
        x = []
        y = []
        range_sum = 0
        for j in range(observation.shape[0]):
            radian = - 179 / 180 * 3.14159 + j * 1 / 180 * 3.14159
            if not (np.isnan(observation[j]) or np.isinf(observation[j])):
                x.append(observation[j] * math.cos(radian))
                y.append(observation[j] * math.sin(radian))
                range_sum += observation[j]

        for j in range(len(x) - 1):
            cx += (x[j] + x[j + 1]) * (x[j] * y[j + 1] - x[j + 1] * y[j])
            cy += (y[j] + y[j + 1]) * (x[j] * y[j + 1] - x[j + 1] * y[j])
            A += 0.5 * (x[j] * y[j + 1] - x[j + 1] * y[j])

        cx /= 6 * A
        cy /= 6 * A
        range_sum /= len(x)
        feat[0] = cx
        feat[1] = cy
        feat[2] = range_sum
        return feat

class Adaptive_Particle_Filter_KD_tree:
    # Initialization
    def __init__(self, global_map, N_particles, paras, Initial_pose=None, Initial_variance=None):
        ## global_map : [H,W] matrix for a occupancy map
        ## N_particles : Number of particles used in the algorithm
        ## paras : other parameters

        ## Initialize map of the environment
        self.global_map = global_map
        self.map_H, self.map_W = self.global_map.shape
        self.scale_factor = math.sqrt(cfg.DSOM_X * cfg.DSOM_Y / (self.map_H * self.map_W))
        self.scaled_X = int(self.map_H * self.scale_factor)
        self.scaled_Y = int(self.map_W * self.scale_factor)
        self.laser_steps = paras['laser_steps']

        ## Initialize iteration
        self.iteration = 0

        ## Initialize particle number
        self.N_particles = N_particles

        self.KDtree_particles = None
        self.x = np.zeros(self.N_particles)
        self.y = np.zeros(self.N_particles)
        self.theta = np.zeros(self.N_particles)
        self.w = np.zeros(self.N_particles)
        self.w_normalized = np.zeros(self.N_particles)

        self.Free_rep = cfg.Free
        free_space = np.where(global_map < cfg.gray_threshold)
        self.free_map = np.ones((self.map_H, self.map_W)) * abs(255 - self.Free_rep)
        self.free_map[free_space] = self.Free_rep
        self.free_map = abs(255 - self.free_map)
        self.free_map = torch.from_numpy(self.free_map).unsqueeze(0).to(cfg.DSOM_device)
        self.map_binary_trans = global_map.copy()
        index = np.where(self.map_binary_trans > cfg.gray_threshold)
        index_ = np.where(self.map_binary_trans <= cfg.gray_threshold)
        self.map_binary_trans[index] = 0
        self.map_binary_trans[index_] = 255

        self.pmap = np.zeros((cfg.DSOM_K_portion, cfg.DSOM_X, cfg.DSOM_Y))
        self.est_w = 0
        self.w_perfect = (paras['laser_portions'] / self.laser_steps) * (
        (scipy.stats.norm.pdf(0, 0, paras['var_rc_weighting_model'])))

        # Init particles
        if Initial_pose == None and Initial_variance == None:
            for i in range(self.N_particles):
                x = np.random.uniform(low=0, high=self.map_H)
                y = np.random.uniform(low=0, high=self.map_W)
                while not (self.global_map[int(x), int(y)] == self.Free_rep):
                    x = np.random.uniform(low=0, high=self.map_H)
                    y = np.random.uniform(low=0, high=self.map_W)
                self.x[i] = x
                self.y[i] = y
                self.theta[i] = np.random.uniform(low=cfg.rad_min, high=cfg.rad_max)

            self.w = np.ones(N_particles) * 1.0 / N_particles
            self.w_normalized = self.w
            self.est_x = np.sum(self.x * self.w_normalized, keepdims=True)
            self.est_y = np.sum(self.y * self.w_normalized, keepdims=True)
            self.est_theta = np.sum(self.theta * self.w_normalized, keepdims=True)

        self.sampling_data = paras['sampling_data']

        print('Building KDtree for sampling......')
        # Building KDtree for particles at previous time step
        self.KDtree_sampling = sklearn.neighbors.KDTree(self.sampling_data[:, 3:6])
        self.sample_num = 5
        print('Finish building KDtree for sampling.')
    # Function for updating particle set
    def update_pf(self, action, observation, paras=None):
        ## action: odometry described as delta_x, delta_y, delta_theta
        ## observation: 2D scan denoted as a vector
        ## paras: parameters for this mode

        # Add up iterations
        self.iteration += 1

        # New particle set
        x_new = np.zeros(self.N_particles)
        y_new = np.zeros(self.N_particles)
        theta_new = np.zeros(self.N_particles)
        w_new = np.zeros(self.N_particles)

        print('DL Aided MCL updating......')

        print('Building KDtree for previous time step......')
        # Building KDtree for particles at previous time step
        previous_set = np.zeros((self.N_particles, 3))

        previous_set[:, 0] = self.x.copy() + np.random.normal(action['delta_x'], action['var_odom_x'], self.N_particles)
        previous_set[:, 1] = self.y.copy() + np.random.normal(action['delta_y'], action['var_odom_y'], self.N_particles)
        previous_set[:, 2] = self.theta.copy() + np.random.normal(action['delta_theta'], action['var_odom_theta'],
                                                                  self.N_particles)
        self.KDtree_particles = sklearn.neighbors.KDTree(previous_set)
        print('Finish building KDtree.')

        print('Updating particle set......')

        # Divide the whole particles set into two parts
        count_mcl = 0
        index_motion = []
        for i in range(self.N_particles):

            if self.iteration == 1:
                p_MCL = 1
            else:
                p_MCL = self.w[i] / self.w_perfect

            if p_MCL > paras['w_cut']:
                p_MCL = 1

            # randomly select update mode
            p_random = np.random.uniform(low=0, high=1)

            if p_random < p_MCL:
                ## Motion model update
                index_motion.append(i)
                count_mcl += 1

        # MCL update
        if count_mcl > 0:
            ## Get the set for MCL update branch
            index_motion = np.array(index_motion)
            x_motion = self.x[index_motion].copy()
            y_motion = self.y[index_motion].copy()
            theta_motion = self.theta[index_motion].copy()
            w_motion = self.w_normalized[index_motion].copy()
            w_motion = w_motion / np.sum(w_motion)
            ## Sample particles based on previous weights
            index_sampled = np.random.choice(count_mcl, count_mcl, p=w_motion)
            x_new[:count_mcl] = x_motion[index_sampled]
            y_new[:count_mcl] = y_motion[index_sampled]
            theta_new[:count_mcl] = theta_motion[index_sampled]
            ## Update newly sampled set based on motion model
            x_new[:count_mcl] += np.random.normal(action['delta_x'], action['var_odom_x'], count_mcl)
            y_new[:count_mcl] += np.random.normal(action['delta_y'], action['var_odom_y'], count_mcl)
            theta_new[:count_mcl] += np.random.normal(action['delta_theta'], action['var_odom_theta'], count_mcl)
            ## Weight these particles with ray casting algorithm
            w_new[:count_mcl] = self.Ray_Casting_Weighting(x_new[:count_mcl], y_new[:count_mcl], theta_new[:count_mcl],
                                                           self.global_map, observation, paras)

        # Dual MCL update
        if count_mcl < self.N_particles:
            # New particle set
            observation_feature = self.ob2feat(observation).reshape((1, -1))
            observation_feature = np.repeat(observation_feature, self.N_particles - count_mcl, axis=0)

            dist, ind = self.KDtree_sampling.query(observation_feature, self.sample_num)

            dist = dist[:,1:]
            ind = ind[:,1:]

            index = np.zeros(self.N_particles - count_mcl)

            dist = 1 / (dist + 0.0000000001)

            for i in range(self.N_particles - count_mcl):
                p = dist[i, :] / np.sum(dist[i, :])
                index_ = np.random.choice(self.sample_num-1, 1, p=p)
                index[i] = ind[i, index_]

            index = index.astype(int)
            x_new[count_mcl:] = self.sampling_data[index, 0]
            y_new[count_mcl:] = self.sampling_data[index, 1]
            theta_new[count_mcl:] = self.sampling_data[index, 2]

            # Sample particle at former time frame
            index = self.find_index_from_previous_set(x_new[count_mcl:], y_new[count_mcl:], theta_new[count_mcl:],
                                                      action)
            index = index.astype(int)
            # Weight new particle
            w_new[count_mcl:] = self.w[index].copy()

        print('Finish updating particle set.')
        print('Update statistics: ')
        print('Motion update: ' + str(count_mcl))
        print('Sampling update: ' + str(self.N_particles - count_mcl))

        # Update particle set
        self.x = x_new
        self.y = y_new

        # Convert orientation into [cfg.rad_min,cfg.rad_max]
        for i in range(self.N_particles):
            while theta_new[i] < cfg.rad_min:
                theta_new[i] += 2 * cfg.pi
            while theta_new[i] > cfg.rad_max:
                theta_new[i] -= 2 * cfg.pi

        self.theta = theta_new
        self.w = w_new
        self.w_normalized = np.true_divide(w_new, np.sum(w_new))

        self.est_x = np.sum(self.x * self.w_normalized, keepdims=True)
        self.est_y = np.sum(self.y * self.w_normalized, keepdims=True)
        self.est_theta = np.sum(self.theta * self.w_normalized, keepdims=True)
        self.est_w = np.sum(
            self.Ray_Casting_Weighting(self.est_x, self.est_y, self.est_theta, self.global_map, observation, paras),
            keepdims=True)

    def Ray_Casting_Weighting(self, x, y, theta, global_map, observation, paras):

        scan_res = paras['scan_res']
        scan_angle_st = paras['scan_angle_st']
        res = paras['resolution']
        max_scan_range = paras['max_scan_range']
        var_rc_weighting_model = paras['var_rc_weighting_model']
        portions = paras['laser_portions']
        w = np.zeros(len(x))

        for index in range(len(x)):

            if np.isnan(x[index]) or np.isnan(y[index]) or x[index] >= self.map_H or x[index] <= 0 or y[
                index] >= self.map_W or y[
                index] <= 0:  # or  not (global_map[int(x[index]),int(y[index])] > cfg.gray_threshold):
                w[index] = 0
            else:
                for p in range(0, portions, self.laser_steps):
                    x_now = x[index].copy()
                    y_now = y[index].copy()
                    theta_now = theta[index].copy() + p * scan_res + scan_angle_st

                    while x_now < self.map_H and x_now > 0 and y_now < self.map_W and y_now > 0 and global_map[
                        int(x_now), int(y_now)] < cfg.gray_threshold:
                        x_axis_base = y_now
                        y_axis_base = self.map_H - x_now
                        x_axis_base += math.cos(theta_now)
                        y_axis_base += math.sin(theta_now)
                        x_now = self.map_H - y_axis_base
                        y_now = x_axis_base

                    rt_len = math.sqrt((x_now - x[index]) ** 2 + (y_now - y[index]) ** 2)

                    if x_now < self.map_H and x_now > 0 and y_now < self.map_W and y_now > 0 and rt_len < max_scan_range / res:
                        rt_v = rt_len
                    else:
                        rt_v = float('inf')

                    if math.isinf(rt_v) and (math.isinf(observation[p]) or math.isnan(observation[p])):
                        p_hit = scipy.stats.norm.pdf(0, 0, var_rc_weighting_model)
                    elif math.isnan(observation[p]) or math.isinf(observation[p]):
                        p_hit = 0
                    else:
                        diff = rt_v - (observation[p] / paras['resolution'])
                        p_hit = scipy.stats.norm.pdf(diff, 0, var_rc_weighting_model)

                    w[index] += p_hit

        return w

    def find_index_from_previous_set(self, x_new, y_new, theta_new, action):

        len_new_set = len(x_new)
        index = np.zeros(len_new_set)

        current_set = np.zeros((len_new_set, 3))
        current_set[:, 0] = x_new
        current_set[:, 1] = y_new
        current_set[:, 2] = theta_new

        dist, ind = self.KDtree_particles.query(current_set, cfg.Num_Sample_from_Previous_Set)

        dist = 1 / (dist + 0.0000000001)

        for i in range(len_new_set):
            p = dist[i, :] / np.sum(dist[i, :])
            index_ = np.random.choice(cfg.Num_Sample_from_Previous_Set, 1, p=p)
            index[i] = ind[i, index_]

        return index

    def rad2dis(self, theta):
        K = 0
        while K < cfg.DSOM_K_portion and self.K_degree[K] < theta:
            K += 1

        return K

    def scan_v2img(self, observation, paras):

        scan_map = np.zeros((self.map_H, self.map_W))

        center_x = self.map_H // 2
        center_y = self.map_W // 2

        scan_res = paras['scan_res']
        scan_angle_st = paras['scan_angle_st']
        portions = paras['laser_portions']

        for p in range(portions):

            if (
            not (math.isinf(observation[p]) or math.isnan(observation[p]) or observation[p] > paras['max_scan_range'])):

                theta_now = p * scan_res + scan_angle_st

                delta_x_axis_base = math.cos(theta_now) * observation[p] / paras['resolution']
                delta_y_axis_base = math.sin(theta_now) * observation[p] / paras['resolution']

                delta_x_axis_map = -1 * delta_y_axis_base
                delta_y_axis_map = delta_x_axis_base

                x_scan = int(center_x + delta_x_axis_map)
                y_scan = int(center_y + delta_y_axis_map)
                if x_scan < self.map_H and y_scan < self.map_W:
                    scan_map[x_scan, y_scan] = 255

        return scan_map

    def Get_Particle_Set(self):

        return self.x, self.y, self.theta, self.w

    def Get_Estimation_Pose(self):

        return self.est_x, self.est_y, self.est_theta, self.est_w

    def ob2feat(self,observation):
        feat = np.zeros(3)
        cx = 0
        cy = 0
        A = 0
        x = []
        y = []
        range_sum = 0
        for j in range(observation.shape[0]):
            radian = - 179 / 180 * 3.14159 + j * 1 / 180 * 3.14159
            if not (np.isnan(observation[j]) or np.isinf(observation[j])):
                x.append(observation[j] * math.cos(radian))
                y.append(observation[j] * math.sin(radian))
                range_sum += observation[j]

        for j in range(len(x) - 1):
            cx += (x[j] + x[j + 1]) * (x[j] * y[j + 1] - x[j + 1] * y[j])
            cy += (y[j] + y[j + 1]) * (x[j] * y[j + 1] - x[j + 1] * y[j])
            A += 0.5 * (x[j] * y[j + 1] - x[j + 1] * y[j])

        cx /= 6 * A
        cy /= 6 * A
        range_sum /= len(x)
        feat[0] = cx
        feat[1] = cy
        feat[2] = range_sum
        return feat

class Mixture_MCL_Particle_Filter_DSOM:
    # Initialization
    def __init__(self, global_map, N_particles, paras, Initial_pose=None, Initial_variance=None):
        ## global_map : [H,W] matrix for a occupancy map
        ## N_particles : Number of particles used in the algorithm
        ## paras : other parameters

        ## Initialize map of the environment
        self.global_map = global_map
        self.map_H, self.map_W = self.global_map.shape
        self.scale_factor = math.sqrt(cfg.DSOM_X * cfg.DSOM_Y / (self.map_H * self.map_W))
        self.scaled_X = int(self.map_H * self.scale_factor)
        self.scaled_Y = int(self.map_W * self.scale_factor)
        self.laser_steps = paras['laser_steps']

        ## Initialize iteration
        self.iteration = 0

        ## Initialize particle number
        self.N_particles = N_particles

        ## Initialize pose generation network
        self.DSOM = DSOM(1, 1)
        self.DSOM.to(cfg.DSOM_device)
        check_point = torch.load(cfg.DSOM_model_path, cfg.DSOM_device)
        self.DSOM.load_state_dict(check_point)
        self.DSOM.eval()
        self.K_degree = np.linspace(cfg.rad_min, cfg.rad_max, cfg.DSOM_K_portion)

        self.Free_rep = cfg.Free
        free_space = np.where(global_map < cfg.gray_threshold)
        self.free_map = np.ones((self.map_H, self.map_W)) * abs(255 - self.Free_rep)
        self.free_map[free_space] = self.Free_rep
        self.free_map = abs(255 - self.free_map)
        self.free_map = torch.from_numpy(self.free_map).unsqueeze(0).to(cfg.DSOM_device)
        self.map_binary_trans = global_map.copy()
        index = np.where(self.map_binary_trans > cfg.gray_threshold)
        index_ = np.where(self.map_binary_trans <= cfg.gray_threshold)
        self.map_binary_trans[index] = 0
        self.map_binary_trans[index_] = 255

        self.pmap = np.zeros((cfg.DSOM_K_portion, cfg.DSOM_X, cfg.DSOM_Y))
        self.est_w = 0
        self.w_perfect = (paras['laser_portions'] / self.laser_steps) * (
        (scipy.stats.norm.pdf(0, 0, paras['var_rc_weighting_model'])))

        self.laser_steps = paras['laser_steps']

        ## Initialize iteration
        self.iteration = 0

        ## Initialize particle number
        self.N_particles = N_particles

        ## Initialize pose generation network
        self.x = np.zeros(self.N_particles)
        self.y = np.zeros(self.N_particles)
        self.theta = np.zeros(self.N_particles)
        self.w_normalized = np.zeros(self.N_particles)
        self.w_perfect = (paras['laser_portions'] / self.laser_steps) * (
        (scipy.stats.norm.pdf(0, 0, paras['var_rc_weighting_model']) * paras['rc_w_hit']))

        # Init particles
        if Initial_pose == None and Initial_variance == None:
            for i in range(self.N_particles):
                x = np.random.uniform(low=0, high=self.map_H)
                y = np.random.uniform(low=0, high=self.map_W)
                while not (self.global_map[int(x), int(y)] == cfg.Free):
                    x = np.random.uniform(low=0, high=self.map_H)
                    y = np.random.uniform(low=0, high=self.map_W)
                self.x[i] = x
                self.y[i] = y
                self.theta[i] = np.random.uniform(low=cfg.rad_min, high=cfg.rad_max)

            self.w_normalized = np.ones(N_particles) * 1.0 / N_particles
            self.est_x = np.sum(self.x * self.w_normalized, keepdims=True)
            self.est_y = np.sum(self.y * self.w_normalized, keepdims=True)
            self.est_theta = np.sum(self.theta * self.w_normalized, keepdims=True)

    # Function for updating particle set
    def update_pf(self, action, observation, paras=None):
        ## action: odometry described as delta_x, delta_y, delta_theta
        ## observation: 2D scan denoted as a vector
        ## paras: parameters for this mode

        # Add up iterations
        self.iteration += 1

        # Convert the scan vector to image [128,128]
        scan_on_image = self.scan_v2img(observation, paras)
        cv2.imwrite(paras['log_dir'] + str(self.iteration) + '_scan.png', scan_on_image)

        # Load the scan and global map to device we use
        scan = cv2.resize(scan_on_image, (self.scaled_Y, self.scaled_X))
        scan = scan[self.scaled_X // 2 - cfg.DSOM_scan_X // 2:self.scaled_X // 2 + cfg.DSOM_scan_X // 2,
               self.scaled_Y // 2 - cfg.DSOM_scan_Y // 2:self.scaled_Y // 2 + cfg.DSOM_scan_Y // 2]
        scan = torch.from_numpy(scan).squeeze().unsqueeze(0).unsqueeze(0).to(cfg.DSOM_device).float()
        g_map = torch.from_numpy(cv2.resize(self.global_map, (self.scaled_Y, self.scaled_X))).squeeze().unsqueeze(
            0).unsqueeze(0).to(cfg.DSOM_device).float()

        # Generate probability map : [32,128,128]
        pmap = torch.exp(self.DSOM(scan, g_map, cfg.DSOM_zero_tensor)).view(1, cfg.DSOM_K_portion, self.scaled_X,
                                                                          self.scaled_Y)

        # Broadcast the pmap to original size
        upsample = torch.nn.Upsample((self.map_H, self.map_W), mode='bilinear')
        pmap = upsample(pmap).squeeze()

        pmap = pmap * self.free_map

        # Normalize pmap
        pmap = pmap / torch.sum(pmap)

        # Convert pmap to an image to show the prediction result
        pmap_ = torch.sum(pmap, dim=0).detach().cpu().numpy()
        cv2.normalize(pmap_, pmap_, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(paras['log_dir'] + str(self.iteration) + '_pmap.png', pmap_)

        pmap = pmap.detach().cpu().numpy()
        self.pmap = pmap
        pmap = pmap.reshape(-1)
        pmap = pmap / np.sum(pmap)

        # randomly select update mode
        p_random = np.random.uniform(low=0, high=1)

        if p_random < paras['P_MCL'] or self.iteration==1:
            # Traditional MCL updating

            # New particle set
            x_new = np.zeros(self.N_particles)
            y_new = np.zeros(self.N_particles)
            theta_new = np.zeros(self.N_particles)

            # Sample particles at former time frame by their weights
            index_sampled = np.random.choice(self.N_particles, self.N_particles, p=self.w_normalized)
            x_new[:] = self.x[index_sampled]
            y_new[:] = self.y[index_sampled]
            theta_new[:] = self.theta[index_sampled]

            # Renew particles based on old particle and action control
            x_new[:] += np.random.normal(action['delta_x'], action['var_odom_x'], self.N_particles)
            y_new[:] += np.random.normal(action['delta_y'], action['var_odom_y'], self.N_particles)
            theta_new[:] += np.random.normal(action['delta_theta'], action['var_odom_theta'], self.N_particles)

            for i in range(self.N_particles):
                while theta_new[i] < cfg.rad_min:
                    theta_new[i] += 2 * cfg.pi
                while theta_new[i] > cfg.rad_max:
                    theta_new[i] -= 2 * cfg.pi

            # Weight new particle
            w_new = self.Ray_Casting_Weighting(x_new, y_new, theta_new, self.global_map, observation, paras)

        else:
            # Dual MCL updating

            # New particle set
            x_new = np.zeros(self.N_particles)
            y_new = np.zeros(self.N_particles)
            theta_new = np.zeros(self.N_particles)
            w_new = np.zeros(self.N_particles)

            if np.sum(pmap) > 0:
                ## Sample particles from pmap
                index = np.random.choice(len(pmap), self.N_particles, p=pmap)
                theta_new[:] = cfg.K_rad[(index // (self.map_H * self.map_W)).astype(int)]
                index = index % (self.map_H * self.map_W)
                x_new[:] = index // self.map_W
                index = index % self.map_W
                y_new[:] = index

            else:
                for i in range(self.N_particles):
                    x = np.random.uniform(low=0, high=self.map_H)
                    y = np.random.uniform(low=0, high=self.map_W)
                    while not (self.global_map[int(x), int(y)] == self.Free_rep):
                        x = np.random.uniform(low=0, high=self.map_H)
                        y = np.random.uniform(low=0, high=self.map_W)
                    x_new[i] = x
                    y_new[i] = y
                    theta_new[i] = np.random.uniform(low=cfg.rad_min, high=cfg.rad_max)
            # Sample particle at former time frame
            index = self.find_index_from_previous_set(x_new[:], y_new[:],
                                                      theta_new[:], action)
            index = index.astype(int)
            # Weight new particle
            w_new[:] = self.w_normalized[index].copy()
        # Update particle set
        if np.sum(w_new) == 0:
            for i in range(self.N_particles):
                x = np.random.uniform(low=0, high=self.map_H)
                y = np.random.uniform(low=0, high=self.map_W)
                while not (self.global_map[int(x), int(y)] == cfg.Free):
                    x = np.random.uniform(low=0, high=self.map_H)
                    y = np.random.uniform(low=0, high=self.map_W)
                self.x[i] = x
                self.y[i] = y
                self.theta[i] = np.random.uniform(low=cfg.rad_min, high=cfg.rad_max)
            self.w_normalized = np.ones(self.N_particles) * 1.0 / self.N_particles
        else:
            self.x = x_new
            self.y = y_new
            self.theta = theta_new
            self.w_normalized = w_new / np.sum(w_new)

        self.est_x = np.sum(self.x * self.w_normalized, keepdims=True)
        self.est_y = np.sum(self.y * self.w_normalized, keepdims=True)
        self.est_theta = np.sum(self.theta * self.w_normalized, keepdims=True)
        self.est_w = np.sum(
            self.Ray_Casting_Weighting(self.est_x, self.est_y, self.est_theta, self.global_map, observation, paras),
            keepdims=True)

    def Ray_Casting_Weighting(self, x, y, theta, global_map, observation, paras):

        scan_res = paras['scan_res']
        scan_angle_st = paras['scan_angle_st']
        max_scan_range = paras['max_scan_range']
        res = paras['resolution']
        var_rc_weighting_model = paras['var_rc_weighting_model']
        portions = paras['laser_portions']
        w = np.zeros(len(x))

        for index in range(len(x)):

            if np.isnan(x[index]) or np.isnan(y[index]) or x[index] >= self.map_H or x[index] <= 0 or y[
                index] >= self.map_W or y[
                index] <= 0 or   (global_map[int(x[index]),int(y[index])] > cfg.gray_threshold):
                w[index] = 0
            else:
                for p in range(0, portions, self.laser_steps):
                    x_now = x[index].copy()
                    y_now = y[index].copy()
                    theta_now = theta[index].copy() + p * scan_res + scan_angle_st

                    while x_now < self.map_H and x_now > 0 and y_now < self.map_W and y_now > 0 and global_map[
                        int(x_now), int(y_now)] < cfg.gray_threshold:
                        x_axis_base = y_now
                        y_axis_base = self.map_H - x_now
                        x_axis_base += math.cos(theta_now)
                        y_axis_base += math.sin(theta_now)
                        x_now = self.map_H - y_axis_base
                        y_now = x_axis_base

                    rt_len = math.sqrt((x_now - x[index]) ** 2 + (y_now - y[index]) ** 2)

                    if x_now < self.map_H and x_now > 0 and y_now < self.map_W and y_now > 0 and rt_len < max_scan_range/res:
                        rt_v = rt_len
                    else:
                        rt_v = float('inf')

                    if math.isinf(rt_v) and (math.isinf(observation[p]) or math.isnan(observation[p])):
                        p_hit = scipy.stats.norm.pdf(0, 0, var_rc_weighting_model)
                    elif math.isnan(observation[p]) or math.isinf(observation[p]):
                        p_hit = 0
                    else:
                        diff = rt_v - (observation[p] / paras['resolution'])
                        p_hit = scipy.stats.norm.pdf(diff, 0, var_rc_weighting_model)

                    p = paras['rc_w_hit'] * p_hit + paras['rc_w_rand'] * 1.0 / (max_scan_range / res)
                    w[index] += p ** 3

        return w

    def find_index_from_previous_set(self, x_new, y_new, theta_new, action):

        print('Building KDtree for previous time step......')
        # Building KDtree for particles at previous time step
        previous_set = np.zeros((self.N_particles, 3))

        previous_set[:, 0] = self.x.copy() + np.random.normal(action['delta_x'], action['var_odom_x'], self.N_particles)
        previous_set[:, 1] = self.y.copy() + np.random.normal(action['delta_y'], action['var_odom_y'], self.N_particles)
        previous_set[:, 2] = self.theta.copy() + np.random.normal(action['delta_theta'], action['var_odom_theta'],
                                                                  self.N_particles)
        KDtree = sklearn.neighbors.KDTree(previous_set)
        print('Finish building KDtree.')

        len_new_set = len(x_new)
        index = np.zeros(len_new_set)

        current_set = np.zeros((len_new_set, 3))
        current_set[:, 0] = x_new
        current_set[:, 1] = y_new
        current_set[:, 2] = theta_new

        dist, ind = KDtree.query(current_set, cfg.Num_Sample_from_Previous_Set)

        dist = 1 / (dist + 0.0000000001)

        for i in range(len_new_set):
            p = dist[i, :] / np.sum(dist[i, :])
            index_ = np.random.choice(cfg.Num_Sample_from_Previous_Set, 1, p=p)
            index[i] = ind[i, index_]

        return index

    def rad2dis(self, theta):
        K = 0
        while K < cfg.DSOM_K_portion and self.K_degree[K] < theta:
            K += 1

        return K

    def scan_v2img(self, observation, paras):

        scan_map = np.zeros((self.map_H, self.map_W))

        center_x = self.map_H // 2
        center_y = self.map_W // 2

        scan_res = paras['scan_res']
        scan_angle_st = paras['scan_angle_st']
        portions = paras['laser_portions']

        for p in range(portions):

            if (not (math.isinf(observation[p]) or math.isnan(observation[p]) or observation[p] > 30)):

                theta_now = p * scan_res + scan_angle_st

                delta_x_axis_base = math.cos(theta_now) * observation[p] / paras['resolution']
                delta_y_axis_base = math.sin(theta_now) * observation[p] / paras['resolution']

                delta_x_axis_map = -1 * delta_y_axis_base
                delta_y_axis_map = delta_x_axis_base

                x_scan = int(center_x + delta_x_axis_map)
                y_scan = int(center_y + delta_y_axis_map)
                if x_scan < self.map_H and y_scan < self.map_W:
                    scan_map[x_scan, y_scan] = 255

        return scan_map

    def Get_Particle_Set(self):

        return self.x, self.y, self.theta, self.w_normalized

    def Get_Estimation_Pose(self):

        return self.est_x, self.est_y, self.est_theta, self.est_w

    def ob2feat(self,observation):
        feat = np.zeros(3)
        cx = 0
        cy = 0
        A = 0
        x = []
        y = []
        range_sum = 0
        for j in range(observation.shape[0]):
            radian = - 179 / 180 * 3.14159 + j * 1 / 180 * 3.14159
            if not (np.isnan(observation[j]) or np.isinf(observation[j])):
                x.append(observation[j] * math.cos(radian))
                y.append(observation[j] * math.sin(radian))
                range_sum += observation[j]

        for j in range(len(x) - 1):
            cx += (x[j] + x[j + 1]) * (x[j] * y[j + 1] - x[j + 1] * y[j])
            cy += (y[j] + y[j + 1]) * (x[j] * y[j + 1] - x[j + 1] * y[j])
            A += 0.5 * (x[j] * y[j + 1] - x[j + 1] * y[j])

        cx /= 6 * A
        cy /= 6 * A
        range_sum /= len(x)
        feat[0] = cx
        feat[1] = cy
        feat[2] = range_sum
        return feat