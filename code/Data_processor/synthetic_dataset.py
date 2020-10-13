import sys
sys.path.append("..")
from config.config import cfg
import numpy as np
import math
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from src.utils import check_collision, ray_trace

if __name__ == "__main__":

    for num in cfg.simulation_Dataset_selected_dataset :
        if num == 1:
            subsection = 'train'
            count = 0
        elif num < 19:
            subsection = 'train'
        elif num == 19 :
            subsection = 'val'
            count = 0
        else:
            subsection = 'val'

        map_env = cv2.imread(cfg.simulation_Dataset_root_path+'map_'+str(num)+'/map.png')
        map_env =cv2.cvtColor(map_env,cv2.COLOR_BGR2GRAY)

        scaled_X = 128
        scaled_Y = 128
        map_env = cv2.resize(map_env,(scaled_Y,scaled_X))

        for m in range(10):

            map_env_with_ob = map_env.copy()

            for i in range(10):
                x = int(np.random.uniform(low=5, high=scaled_X-5))
                y = int(np.random.uniform(low=5, high=scaled_Y-5))
                cv2.rectangle(map_env_with_ob,(y,x),(y+2,x+2),(255),-1)
                x = int(np.random.uniform(low=5, high=scaled_X-5))
                y = int(np.random.uniform(low=5, high=scaled_Y-5))
                cv2.circle(map_env_with_ob,(y,x),2,(255),-1,0)

            cv2.imwrite(cfg.simulation_Dataset_root_path+'map_'+str(num)+'/map_ob_'+str(m)+'.png',map_env_with_ob)
            cv2.imwrite(cfg.simulation_Dataset_root_path+'map_'+str(num)+'/map_.png',map_env)
            # Preprocess the map to [W,H]

            # Generate data
            for i in range(3,scaled_X-3,3):
                for j in range(3,scaled_Y-3,3):

                    if check_collision(i,j,map_env_with_ob):
                        pmap = np.zeros((scaled_X,scaled_Y))
                        pmap[i,j] = 255
                        pmap = cv2.GaussianBlur(pmap, (7, 7), 3)
                        cv2.normalize(pmap,pmap,0,255,cv2.NORM_MINMAX)

                        for k in range(10):
                            degree = np.random.uniform(low=cfg.rad_min, high=cfg.rad_max)
                            scan = ray_trace(i,j,map_env_with_ob,degree,cfg.simulation_Dataset_max_range*cfg.simulation_Dataset_resolution,cfg.simulation_Dataset_st_angle,cfg.simulation_Dataset_scan_steps,cfg.simulation_Dataset_angle_step)

                            if len(np.where(scan==0)) > 0:

                                cv2.imwrite(cfg.simulation_Dataset_root_path+'Data_Transformed/2/'+subsection+'/scan/'+str(count)+'.png',scan)
                                cv2.imwrite(cfg.simulation_Dataset_root_path+'Data_Transformed/2/'+subsection+'/pmap/'+str(count)+'.png',pmap)
                                cv2.imwrite(cfg.simulation_Dataset_root_path+'Data_Transformed/2/'+subsection+'/map/'+str(count)+'.png',map_env)
                                np.savetxt(cfg.simulation_Dataset_root_path+'Data_Transformed/2/'+subsection+'/rotation/'+str(count)+'.txt',np.array([degree]))

                                count += 1

                                print(subsection+':'+str(count))
                        sys.exit()
            

