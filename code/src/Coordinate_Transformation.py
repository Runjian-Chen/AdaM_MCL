import numpy as np
import math
##############
#
#   Coordinate_Transformation: transformation between three coordinates (original, base, image) in meters.
#
##############

class Coordinate_Transformation():
    def __init__(self, x_max, x_min, y_max, y_min):

        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min

    def ori2base(self,point):
        x_ori , y_ori = point
        x_base = x_ori - self.x_min
        y_base = y_ori - self.y_min
        return (x_base,y_base)

    def base2img(self,point):
        x_base, y_base = point
        x_img = self.y_max - self.y_min - y_base
        y_img = x_base
        return (x_img, y_img)

    def ori2img(self,point):
        return self.base2img(self.ori2base(point))

    def img2base(self,point):
        x_img, y_img = point
        x_base = y_img
        y_base = self.y_max - self.y_min - x_img
        return (x_base, y_base)

    def base2ori(self,point):
        x_base, y_base = point
        x_ori = x_base + self.x_min
        y_ori = y_base + self.y_min
        return (x_ori, y_ori)

    def img2ori(self,point):
        return self.base2ori(self.img2base(point))

    def get_env_size(self):
        return (self.y_max-self.y_min+1,self.x_max-self.x_min+1)

    def img2array(self,point,res):
        x_img , y_img = point
        return (int(x_img/res),int(y_img/res))

    def ori2array(self,point,res):
        return self.img2array(self.ori2img(point),res)

    def Bicocca_trajectory2ori(self,pose):
        x_traj , y_traj , theta_traj = pose
        delta_theta = 3.14159 / 2 - 0.005
        delta_x = -0.7
        delta_y = -7.6
        x_ori = x_traj * math.cos(delta_theta) - y_traj * math.sin(delta_theta)
        y_ori = x_traj * math.sin(delta_theta) + y_traj * math.cos(delta_theta)
        x_ori += delta_x
        y_ori += delta_y
        theta_ori = theta_traj + delta_theta
        return (x_ori,y_ori,theta_ori)