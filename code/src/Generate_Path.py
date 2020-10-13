import sys
sys.path.append("..")
import cv2
import numpy as np
import math
from config.config import cfg
import os

def A_star(env,st,des):
    kernel = np.ones((5, 5), np.uint8)
    maps = cv2.dilate(env, kernel, iterations=1)
    informap = cv2.cvtColor(env, cv2.COLOR_GRAY2BGR)
    maps_size = np.array(maps)  # 获取图像行和列大小
    hight = maps_size.shape[0]  # 行数->y
    width = maps_size.shape[1]  # 列数->x

    star = {'位置': st, '代价': 700, '父节点': st}  # 起点
    end = {'位置': des, '代价': 0, '父节点': des}  # 终点

    openlist = []  # open列表，存储可能路径
    closelist = [star]  # close列表，已走过路径
    step_size = 1
    iter = 0
    while iter < 2000:
        s_point = closelist[-1]['位置']  # 获取close列表最后一个点位置，S点
        add = ([0, step_size], [0, -step_size], [step_size, 0], [-step_size, 0])  # 可能运动的四个方向增量
        for i in range(len(add)):
            x = s_point[0] + add[i][0]  # 检索超出图像大小范围则跳过
            if x < 0 or x >= width:
                continue
            y = s_point[1] + add[i][1]
            if y < 0 or y >= hight:  # 检索超出图像大小范围则跳过
                continue
            G = abs(x - star['位置'][0]) + abs(y - star['位置'][1])  # 计算代价
            H = abs(x - end['位置'][0]) + abs(y - end['位置'][1])  # 计算代价
            F = G + H
            if H < 20:  # 当逐渐靠近终点时，搜索的步长变小
                step_size = 1
            addpoint = {'位置': (x, y), '代价': F, '父节点': s_point}  # 更新位置
            count = 0
            for i in openlist:
                if i['位置'] == addpoint['位置']:
                    count += 1
            for i in closelist:
                if i['位置'] == addpoint['位置']:
                    count += 1
            if count == 0:  # 新增点不在open和close列表中
                if maps[int(x), int(y)] == 0:  # 非障碍物
                    openlist.append(addpoint)
        t_point = {'位置': (50, 50), '代价': 10000, '父节点': (50, 50)}
        for j in range(len(openlist)):  # 寻找代价最小点
            if openlist[j]['代价'] < t_point['代价']:
                t_point = openlist[j]
        for j in range(len(openlist)):  # 在open列表中删除t点
            if t_point == openlist[j]:
                openlist.pop(j)
                break
        closelist.append(t_point)  # 在close列表中加入t点
        # cv2.circle(informap,t_point['位置'],1,(200,0,0),-1)
        if t_point['位置'] == end['位置']:  # 找到终点！！
            print("找到终点")
            break
        iter += 1
    # print(closelist)
    if iter == 2000:
        return []
    # 逆向搜索找到路径
    road = []
    road.append(closelist[-1])
    point = road[-1]
    k = 0

    iter = 0
    while 1:
        for i in closelist:
            if i['位置'] == point['父节点']:  # 找到父节点
                point = i
                # print(point)
                road.append(point)

        if point == star:
            print("路径搜索完成")
            break

        iter += 1
        if iter > 100:
            return []

    return road


if __name__ == "__main__":
    save_path = '../../Dataset/simulation/Data_Transformed/Test_MCL/'
    gt_poses = []
    obs = []
    for i in range(10):
        env = cv2.imread(save_path + 'global_map.png')
        env = cv2.cvtColor(env, cv2.COLOR_BGR2GRAY)
        H , W = env.shape[:2]
        # Random obstacles
        for ob in range(20):
            x = int(np.random.uniform(low=5, high=H - 5))
            y = int(np.random.uniform(low=5, high=W - 5))
            cv2.rectangle(env, (y, x), (y + 1, x + 1), (255), -1)
            x = int(np.random.uniform(low=5, high=H - 5))
            y = int(np.random.uniform(low=5, high=W - 5))
            cv2.circle(env, (y, x), 1, (255), -1, 0)

        paths = []
        while len(paths)<2:
            # Random starting point and ending point
            x = np.random.uniform(low=0, high=H)
            y = np.random.uniform(low=0, high=W)
            while not ((env[int(x), int(y)] == cfg.Free)):
                x = np.random.uniform(low=0, high=H)
                y = np.random.uniform(low=0, high=W)
            st_x = int(x)
            st_y = int(y)

            x = np.random.uniform(low=0, high=H)
            y = np.random.uniform(low=0, high=W)
            while not ((env[int(x), int(y)] == cfg.Free) and (st_x-x)**2+(st_y-y)**2<2000 and (st_x-x)**2+(st_y-y)**2>1600):
                x = np.random.uniform(low=0, high=H)
                y = np.random.uniform(low=0, high=W)
            end_x = int(x)
            end_y = int(y)

            print((st_x,st_y))
            print((end_x,end_y))
            print('Planning: '+str(i))
            path = A_star(env,(st_x,st_y),(end_x,end_y))

            if len(path)>0:
                paths.append(path)

        ob = []
        gt_pose = []
        odo = []
        env_path = cv2.cvtColor(env, cv2.COLOR_GRAY2BGR)
        for p in range(2):
            for step in range(len(paths[p])-1):
                x , y = paths[p][step]['位置']
                x_next , y_next = paths[p][step+1]['位置']
                x_axis_base = y
                y_axis_base = H - x
                x_axis_base_next = y_next
                y_axis_base_next = H - x_next
                delta_x_base = x_axis_base_next - x_axis_base
                delta_y_base = y_axis_base_next - y_axis_base
                theta = math.atan2(delta_y_base,delta_x_base)
                gt_pose.append(np.array([x,y,theta]))
                scan = np.zeros(360)
                cv2.circle(env_path, (y, x), 1, (0, 0, 200), -1)
                #cv2.imshow('path', env_path)
                #cv2.waitKey()
                for angle in range(360):
                    x_axis_map = x
                    y_axis_map = y
                    radian = theta + - 179 / 180 * 3.14159 + angle * 1 / 180 * 3.14159

                    while int(x_axis_map) > 0 and int(x_axis_map) < H - 1 and int(y_axis_map) > 0 and int(
                            y_axis_map) < W - 1 and (env[int(x_axis_map), int(y_axis_map)] <= 230):
                        x_axis_base = y_axis_map
                        y_axis_base = H - x_axis_map
                        x_axis_base += math.cos(radian)
                        y_axis_base += math.sin(radian)
                        x_axis_map = H - y_axis_base
                        y_axis_map = x_axis_base

                    if int(x_axis_map) > 0 and int(x_axis_map) < H - 1 and int(y_axis_map) > 0 and int(
                            y_axis_map) < W - 1 and (
                            x - x_axis_map) ** 2 + (y - y_axis_map) ** 2 < 900:
                        scan[angle] = math.sqrt((x - x_axis_map) ** 2 + (y - y_axis_map) ** 2)
                    else:
                        scan[angle] = math.nan

                ob.append(scan)


        for step in range(len(paths[0]) - 2):
            delta_x , delta_y , delta_theta = gt_pose[step+1]-gt_pose[step]
            delta_x += np.random.normal(0,0.1)
            delta_y += np.random.normal(0,0.1)
            delta_theta += np.random.normal(0, 0.1)
            odo.append(np.array([delta_x , delta_y , delta_theta]))
        odo.append(np.array([0,0,0]))
        for step in range(len(paths[0]) - 1, len(paths[0]) - 1 + len(paths[1]) - 1 - 1):
            delta_x , delta_y , delta_theta = gt_pose[step+1]-gt_pose[step]
            delta_x += np.random.normal(0,0.1)
            delta_y += np.random.normal(0,0.1)
            delta_theta += np.random.normal(0, 0.1)
            odo.append(np.array([delta_x , delta_y , delta_theta]))

        odo = np.array(odo)
        gt_pose = np.array(gt_pose)
        ob = np.array(ob)
        gt_poses.append(gt_pose)
        obs.append(ob)
        if not os.path.exists(save_path + str(i)):
            os.makedirs(save_path + str(i))
        np.savetxt(save_path + str(i) + '/gt_pose.txt', gt_pose)
        np.savetxt(save_path + str(i) + '/odo.txt', odo)
        np.savetxt(save_path + str(i) + '/laser.txt', ob)
        env_ori = cv2.imread(save_path + 'global_map.png')
        cv2.imwrite(save_path + str(i) + '/global_map.png',env_ori)
        cv2.imwrite(save_path + str(i) + '/global_map_ob.png', env_path)


'''
print(x)
print(y)
print(x_next)
print(y_next)
print(theta)
cv2.circle(env_path, (y,x), 1, (0, 0, 200), -1)
cv2.imshow('path',env_path)
cv2.waitKey()
'''








