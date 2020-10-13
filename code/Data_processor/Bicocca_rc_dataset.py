import numpy as np
import dxfgrabber
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append("..")
import cv2
from config.config import cfg
from src.Coordinate_Transformation import Coordinate_Transformation
import math
import argparse
from src.utils import check_collision, ray_trace
import os

def getMinXY(shapes):
    minX, minY = 99999, 99999
    for shape in shapes:
        if shape.dxftype == 'LINE':
            minX = min(minX, shape.start[0], shape.end[0])
            minY = min(minY, shape.start[1], shape.end[1])
        elif shape.dxftype == 'ARC':
            minX = min(minX, shape.center[0])
            minY = min(minY, shape.center[1])

    return minX, minY

def getMaxXY(shapes):
    maxX, maxY = -99999, -99999
    for shape in shapes:
        if shape.dxftype == 'LINE':
            maxX = max(maxX, shape.start[0], shape.end[0])
            maxY = max(maxY, shape.start[1], shape.end[1])
        elif shape.dxftype == 'ARC':
            maxX = max(maxX, shape.center[0])
            maxY = max(maxY, shape.center[1])

    return maxX, maxY


def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)

    for i in range(len_contour):
        if contours[i].shape[0] < 800:
            mask = cv2.drawContours(mask, contours, i, (255, 255, 255), -1)

    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--Generate_RC_data', dest='Generate_RC_data', action='store_true')
    flag_parser.add_argument('--no-Generate_RC_data', dest='Generate_RC_data', action='store_false')
    parser.set_defaults(Generate_RC_data=True)
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--Testing', dest='Testing', action='store_true')
    flag_parser.add_argument('--no-Testing', dest='Testing', action='store_false')
    parser.set_defaults(Testing=False)
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--Generate_GT_data', dest='Generate_GT_data', action='store_true')
    flag_parser.add_argument('--no-Generate_GT_data', dest='Generate_GT_data', action='store_false')
    parser.set_defaults(Testing=False)
    args = parser.parse_args()

    """
        Generate image of environment with dxf file.
    """
    Bicocca_dxf = dxfgrabber.readfile(cfg.Bicocca_dxf_path)
    shapes = Bicocca_dxf.entities.get_entities()
    x_min, y_min = getMinXY(shapes)
    x_max, y_max = getMaxXY(shapes)
    Bicocca_Coordinate_Transformation = Coordinate_Transformation(x_max, x_min, y_max, y_min)

    shapes = Bicocca_dxf.entities.get_entities()
    env_size = Bicocca_Coordinate_Transformation.get_env_size()
    env_size = Bicocca_Coordinate_Transformation.img2array(env_size,cfg.Bicocca_resolution)

    env = np.zeros((env_size[0],env_size[1],3), np.uint8)
    for shape in shapes:
        if shape.dxftype == 'LINE':
            start = shape.start[0:2]
            end = shape.end[0:2]
            x_start, y_start = Bicocca_Coordinate_Transformation.ori2array(start,cfg.Bicocca_resolution)
            x_end, y_end = Bicocca_Coordinate_Transformation.ori2array(end,cfg.Bicocca_resolution)

            env = cv2.line(env, (y_start,x_start), (y_end,x_end), (255,255,255), 1)
        elif shape.dxftype == 'ARC':
            center = shape.center
            x_center , y_center = Bicocca_Coordinate_Transformation.ori2array(center,cfg.Bicocca_resolution)
            radius = int(shape.radius / cfg.Bicocca_resolution)
            if (shape.start_angle > 180) and (shape.end_angle < shape.start_angle):
                env = cv2.ellipse(env, (y_center,x_center), (radius, radius), 180, int(shape.start_angle) - 180, 180 + int(shape.end_angle), (255,255,255), 1)
            else:
                env = cv2.ellipse(env, (y_center,x_center), (radius, radius), 0, int(360 - shape.start_angle), int(360 - shape.end_angle), (255,255,255), 1)

        elif shape.dxftype == 'POLYLINE':
            points = shape.points
            for i in range(len(points)-1):
                start = points[i][0:2]
                end = points[i+1][0:2]
                x_start, y_start = Bicocca_Coordinate_Transformation.ori2array(start, cfg.Bicocca_resolution)
                x_end, y_end = Bicocca_Coordinate_Transformation.ori2array(end, cfg.Bicocca_resolution)

            env = cv2.line(env, (y_start, x_start), (y_end, x_end), (255,255,255), 1)
    env = cv2.cvtColor(env, cv2.COLOR_BGR2GRAY)
    env = FillHole(env)
    env = cv2.cvtColor(env, cv2.COLOR_GRAY2BGR)
    """
        If we do not want to generate ray casting data set
    """
    if args.Testing:
        """
            Map laser scan onto image with ground truth pose.
        """
        Bicocca_gt_trajectory = np.loadtxt(cfg.Bicocca_gt_trajectory_path,
                                           delimiter=',')
        Bicocca_laser_scan = np.loadtxt(cfg.Bicocca_laser_scan_path,
                                        delimiter=',')
        index_laser = 0
        x_img_start, y_img_start, x_img_end, y_img_end = cfg.Bicocca_crop[0]
        for i in range(Bicocca_gt_trajectory.shape[0]-1):
            env_ = env.copy()
            time_stamp_now = Bicocca_gt_trajectory[i, 0]
            time_stamp_next = Bicocca_gt_trajectory[i + 1, 0]
            while (Bicocca_laser_scan[index_laser, 0] < time_stamp_now):
                index_laser += 1
            time_stamp_laser = Bicocca_laser_scan[index_laser, 0]

            pose_now = Bicocca_gt_trajectory[i,1:]
            x_now , y_now, theta_now = Bicocca_Coordinate_Transformation.Bicocca_trajectory2ori(pose_now)
            x_now , y_now= Bicocca_Coordinate_Transformation.ori2array((x_now , y_now), cfg.Bicocca_resolution)

            pose_next = Bicocca_gt_trajectory[i+1, 1:]
            x_next, y_next, theta_next = Bicocca_Coordinate_Transformation.Bicocca_trajectory2ori(pose_next)
            x_next, y_next = Bicocca_Coordinate_Transformation.ori2array((x_next, y_next), cfg.Bicocca_resolution)

            x = x_now + (x_next - x_now) / (time_stamp_next - time_stamp_now) * (time_stamp_laser - time_stamp_now)
            y = y_now + (y_next - y_now) / (time_stamp_next - time_stamp_now) * (time_stamp_laser - time_stamp_now)
            theta = theta_now + (theta_next - theta_now) / (time_stamp_next - time_stamp_now) * (time_stamp_laser - time_stamp_now)

            laser_scan = Bicocca_laser_scan[index_laser,1:]

            for j in range(len(laser_scan)):
                if laser_scan[j] < 30:
                    scan_theta_now = cfg.Bicocca_st_angle + j * cfg.Bicocca_angle_step
                    delta_x_axis_base = math.cos(scan_theta_now) * laser_scan[j] / cfg.Bicocca_resolution
                    delta_y_axis_base = math.sin(scan_theta_now) * laser_scan[j] / cfg.Bicocca_resolution
                    delta_x_axis_base_rot = delta_x_axis_base * math.cos(theta) - delta_y_axis_base * math.sin(theta)
                    delta_y_axis_base_rot = delta_x_axis_base * math.sin(theta) + delta_y_axis_base * math.cos(theta)

                    delta_x_axis_map = -1 * delta_y_axis_base_rot
                    delta_y_axis_map = delta_x_axis_base_rot

                    x_scan = int(x + delta_x_axis_map)
                    y_scan = int(y + delta_y_axis_map)

                    env_[x_scan, y_scan, 2] = 255
                    env_[x_scan, y_scan, 1] = 0
                    env_[x_scan, y_scan, 0] = 255

            if x > x_img_start and x < x_img_end and y > y_img_start and y < y_img_end:

                cv2.imwrite('/Users/chenrj/Desktop/1.png',env_)
                cv2.imshow('scan',env_)
                cv2.waitKey()
    elif args.Generate_RC_data:
        """
            Generate ray casting data set with the map.
            * First crop the map into submaps.
            * Generate ray casting data on submaps.
        """
        for i in range(len(cfg.Bicocca_crop)):
            env_crop = env.copy()
            x_img_start, y_img_start, x_img_end, y_img_end = cfg.Bicocca_crop[i]
            env_crop = env_crop[x_img_start:x_img_end,y_img_start:y_img_end]
            path = cfg.Bicocca_dataset_root_path + 'Data_Transformed_'+ cfg.Bicocca_dataset + '/' + str(i+1)
            map_env = env_crop

            count = 0

            for p in range(20):
                map_env_with_ob = map_env.copy()
                X , Y = map_env_with_ob.shape

                for j in range(60):
                    x = int(np.random.uniform(low=5, high=X - 5))
                    y = int(np.random.uniform(low=5, high=Y - 5))
                    cv2.rectangle(map_env_with_ob, (y, x), (y + 1, x + 1), (255), -1)
                    x = int(np.random.uniform(low=5, high=X - 5))
                    y = int(np.random.uniform(low=5, high=Y - 5))
                    cv2.circle(map_env_with_ob, (y, x), 1, (255), -1, 0)

                cv2.imwrite(path + '/map_ob_'+str(p+1)+'.png', map_env_with_ob)
                cv2.imwrite(path + '/map_.png', map_env)

                for XX in range(5, X-5, 2):
                    for YY in range(5, Y-5, 2):

                        if check_collision(XX, YY, map_env_with_ob):
                            pmap = np.zeros((X, Y))
                            pmap[XX, YY] = 255
                            pmap = cv2.GaussianBlur(pmap, (7, 7), 3)
                            cv2.normalize(pmap, pmap, 0, 255, cv2.NORM_MINMAX)

                            for k in range(10):
                                degree = np.random.uniform(low=cfg.rad_min, high=cfg.rad_max)
                                scan = ray_trace(XX, YY, map_env_with_ob, degree, cfg.Bicocca_laser_max_range / cfg.Bicocca_resolution,cfg.Bicocca_st_angle,cfg.Bicocca_scan_steps,cfg.Bicocca_angle_step)

                                if len(np.where(scan == 255)) > 0:
                                    cv2.imwrite(path + '/scan/' + str(count) + '.png', scan)
                                    cv2.imwrite(path + '/pmap/' + str(count) + '.png', pmap)
                                    cv2.imwrite(path + '/map/' + str(count) + '.png', map_env)
                                    np.savetxt(path + '/rotation/' + str(count) + '.txt', np.array([degree]))

                                    count += 1

                                    print(str(p+1) + 'th scene ---- ' + str(count) + ' in total')
    elif args.Generate_GT_data:
        """
            Generate ground truth data (odometry, observation and pose) with the map.
            * First crop the map into submaps.
            * Generate ray casting data on submaps.
        """
        Bicocca_gt_trajectory = np.loadtxt(cfg.Bicocca_gt_trajectory_path,
                                           delimiter=',')
        Bicocca_laser_scan = np.loadtxt(cfg.Bicocca_laser_scan_path,
                                        delimiter=',')

        for i in range(len(cfg.Bicocca_crop)):
            env_crop = env.copy()
            x_img_start, y_img_start, x_img_end, y_img_end = cfg.Bicocca_crop[i]
            env_crop = env_crop[x_img_start:x_img_end, y_img_start:y_img_end]
            gt_pose=[]
            laser=[]
            odo = []
            for j in range(Bicocca_gt_trajectory.shape[0]-1):
                index_laser = 0
                time_stamp_now = Bicocca_gt_trajectory[j, 0]
                time_stamp_next = Bicocca_gt_trajectory[j + 1, 0]
                pose_now = Bicocca_gt_trajectory[j, 1:]
                x_now, y_now, theta_now = Bicocca_Coordinate_Transformation.Bicocca_trajectory2ori(pose_now)
                x_now, y_now = Bicocca_Coordinate_Transformation.ori2array((x_now, y_now), cfg.Bicocca_resolution)
                pose_next = Bicocca_gt_trajectory[j + 1, 1:]
                x_next, y_next, theta_next = Bicocca_Coordinate_Transformation.Bicocca_trajectory2ori(pose_next)
                x_next, y_next = Bicocca_Coordinate_Transformation.ori2array((x_next, y_next),
                                                                             cfg.Bicocca_resolution)

                if x_now > x_img_start and x_now < x_img_end and y_now > y_img_start and y_now < y_img_end \
                        and  x_next > x_img_start and x_next < x_img_end and y_next > y_img_start and y_next < y_img_end:

                    while (Bicocca_laser_scan[index_laser, 0] < time_stamp_now):
                        index_laser += 1
                    time_stamp_laser = Bicocca_laser_scan[index_laser, 0]

                    x = x_now + (x_next - x_now) / (time_stamp_next - time_stamp_now) * (
                                time_stamp_laser - time_stamp_now)
                    y = y_now + (y_next - y_now) / (time_stamp_next - time_stamp_now) * (
                                time_stamp_laser - time_stamp_now)
                    theta = theta_now + (theta_next - theta_now) / (time_stamp_next - time_stamp_now) * (
                                time_stamp_laser - time_stamp_now)

                    laser_scan = Bicocca_laser_scan[index_laser, 1:]
                    count_laser_inside = 0
                    for j in range(len(laser_scan)):
                        if laser_scan[j] < 30:
                            scan_theta_now = cfg.Bicocca_st_angle + j * cfg.Bicocca_angle_step
                            delta_x_axis_base = math.cos(scan_theta_now) * laser_scan[j] / cfg.Bicocca_resolution
                            delta_y_axis_base = math.sin(scan_theta_now) * laser_scan[j] / cfg.Bicocca_resolution
                            delta_x_axis_base_rot = delta_x_axis_base * math.cos(theta) - delta_y_axis_base * math.sin(
                                theta)
                            delta_y_axis_base_rot = delta_x_axis_base * math.sin(theta) + delta_y_axis_base * math.cos(
                                theta)

                            delta_x_axis_map = -1 * delta_y_axis_base_rot
                            delta_y_axis_map = delta_x_axis_base_rot

                            x_scan = int(x + delta_x_axis_map)
                            y_scan = int(y + delta_y_axis_map)

                            if x_scan > x_img_start and x_scan < x_img_end and y_scan > y_img_start and y_scan < y_img_end:
                                count_laser_inside += 1
                            else:
                                laser_scan[j] = math.nan
                        else:
                            laser_scan[j] = math.nan
                    x = x - x_img_start
                    y = y - y_img_start
                    if count_laser_inside > 10:
                        laser.append(laser_scan)
                        gt_pose.append((x,y,theta))
                        if len(gt_pose)>1:
                            odo.append(np.array(gt_pose[-1])-np.array(gt_pose[-2]))

            laser = np.array(laser)
            gt_pose = np.array(gt_pose)
            odo = np.array(odo)
            path = cfg.Bicocca_dataset_root_path + 'Data_Transformed_' + cfg.Bicocca_dataset + '/Test_MCL/' + str(i + 1) + '/'
            os.makedirs(path,exist_ok=True)
            np.savetxt(path + 'laser.txt', laser)
            np.savetxt(path + 'gt_pose.txt', gt_pose)
            np.savetxt(path + 'odo.txt', odo)
            cv2.imwrite(path + 'global_map.png',env_crop)