import sys
sys.path.append("..")
from config.config import cfg
import numpy as np
import math
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

def check_collision(x,y,map_env):

  H , W = map_env.shape
  for i in range(-4,4):
    for j in range(-4,4):
      xx = x + i
      yy = y + j
      if int(xx) == 0 or int(xx) == H-1 or int(yy) == 0 or int(yy) == W-1 or map_env[xx,yy] > cfg.gray_threshold:
        return False
  
  return True

def ray_trace(x,y,map_env,degree):

  H , W = map_env.shape
  scan = np.zeros((H,W))
  range_ = (cfg.UPO_Dataset_range_max * 5.6) ** 2

  for i in range(cfg.UPO_Dataset_angle_portions):
    
    x_axis_map = x
    y_axis_map = y
    radian = degree + cfg.UPO_Dataset_angle_min + i * cfg.UPO_Dataset_angle_increment
   
    while int(x_axis_map)>0 and int(x_axis_map)<H and int(y_axis_map)>0 and int(y_axis_map)<W and (not (map_env[int(x_axis_map),int(y_axis_map)] == 255)):
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
      
      scan[int(H//2+delta_x_axis_map),int(W//2+delta_y_axis_map)] = 255

  return scan


if __name__ == "__main__":

    map_env = cv2.imread(cfg.UPO_Dataset_map_path)
    map_env =cv2.cvtColor(map_env,cv2.COLOR_BGR2GRAY)
    H , W = map_env.shape
    scaled_X , scaled_Y = map_env.shape

    total_ = 0
    
    for num in range(10):
        
        map_env_with_ob = map_env.copy()
        
        for i in range(60):
            x = int(np.random.uniform(low=5, high=scaled_X-5))
            y = int(np.random.uniform(low=5, high=scaled_Y-5))
            cv2.rectangle(map_env_with_ob,(x,y),(x+2,y+2),(255),-1)
            x = int(np.random.uniform(low=5, high=scaled_X-5))
            y = int(np.random.uniform(low=5, high=scaled_Y-5))
            cv2.circle(map_env_with_ob,(x,y),1,(255),-1,0)
            
        
        #cv2.imwrite(cfg.UPO_Dataset_root_path+'rc/'+str(num)+'map_ob.png',map_env_with_ob)
        #cv2.imwrite(cfg.UPO_Dataset_root_path+'rc/'+str(num)+'map.png',map_env)
        # Preprocess the map to [W,H]

        # Generate data
        count = 0
        
        for i in range(0,scaled_X,3):
            for j in range(0,scaled_Y,3):
                
                if check_collision(i,j,map_env_with_ob):
                    pmap = np.zeros((scaled_X,scaled_Y))
                    pmap[i,j] = 255
                    pmap = cv2.GaussianBlur(pmap, (7, 7), 3) 
                    cv2.normalize(pmap,pmap,0,255,cv2.NORM_MINMAX)
                    
                    for k in range(5):
                        degree = np.random.uniform(low=cfg.rad_min, high=cfg.rad_max)
                        
                        scan = ray_trace(i,j,map_env_with_ob,degree)

                        if len(np.where(scan==0)) > 0:

                            cv2.imwrite(cfg.UPO_Dataset_root_path+'Data_Transformed/1/train/scan/'+str(total_)+'.png',scan)
                            cv2.imwrite(cfg.UPO_Dataset_root_path+'Data_Transformed/1/train/pmap/'+str(total_)+'.png',pmap)
                            cv2.imwrite(cfg.UPO_Dataset_root_path+'Data_Transformed/1/train/map/'+str(total_)+'.png',map_env)
                            np.savetxt(cfg.UPO_Dataset_root_path+'Data_Transformed/1/train/rotation/'+str(total_)+'.txt',np.array([degree]))

                            total_ += 1
                            count += 1

                            print(str(num)+':'+str(count))



