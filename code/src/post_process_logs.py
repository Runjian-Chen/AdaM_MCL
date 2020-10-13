import os
import sys
import numpy as np
import math
from config.config import cfg
from src.utils import blue , green
import scipy.stats as sta
import matplotlib.pyplot as plt
import pandas as pd

styles = ['red','green','blue','cyan','yellow','orange','purple','gold','brown','gray']
colors = ['red','green','blue','cyan','yellow','orange','purple','gold','brown','gray']

def post_process_logs(log_dir,T_pos,T_rot,T_step):
    kidnapping_points = {'simulation_1': 60 , 'simulation_2': 70 , 'simulation_3': 75 , 'simulation_4': 90 , 'simulation_5': 60 , 'simulation_6': 80 , 'UPO_2':[100,200],
                         'Bicocca_1':100 , 'Bicocca_2':100
                 }
    datasets = os.listdir(log_dir)
    datasets = ['simulation']
    for dataset in datasets:
        if not dataset.startswith('.'):
            dataset_dir = log_dir + dataset + '/'
            subdatasets = os.listdir(dataset_dir)
            subdatasets = ['6']
            save_results = open(dataset_dir + 'results.txt', mode='a+')
            for subdataset in subdatasets:
                if not subdataset.startswith('.'):
                    #kidnapping_point = kidnapping_points[dataset+'_'+subdataset]
                    sub_dataset_dir = dataset_dir + subdataset + '/'
                    methods = os.listdir(sub_dataset_dir)
                    methods = ['MCL_Particle_Filter','Mixture_MCL_Particle_Filter_test','DL_Aided_Particle_Filter']
                    color = ['green','blue','red']
                    style = ['green','blue','red']
                    labels = ['MCL','Mixture MCL','Adaptive Mixture MCL']
                    #methods.sort(key=lambda x: float(x[-3:]))
                    methods_error_pos = []
                    methods_error_rot = []
                    methods_CI_pos = []
                    methods_CI_rot = []
                    methods_used = []
                    error_pos_gls = []
                    error_rot_gls = []
                    step_gls = []
                    error_pos_kns = []
                    error_rot_kns = []
                    step_kns = []
                    for method in methods:
                        if not (method.startswith('.') or method.startswith('g')):
                            methods_used.append(method)
                            method_dir = sub_dataset_dir + method + '/'
                            exps = os.listdir(method_dir)
                            error_evolution_pos = []
                            error_evolution_rot = []
                            error_pos_gl = []
                            error_rot_gl = []
                            conv_step_gl = []
                            error_pos_kn = []
                            error_rot_kn = []
                            conv_step_kn = []
                            exp_num_gl = 0
                            exp_num_kn = 0

                            for exp in exps:
                                if exp.startswith('log') and os.path.exists(method_dir+exp+'/particle_set.npy'):
                                    exp_num_gl += 1
                                    exp_num_kn += 1
                                    #gt_pose = np.loadtxt('../Dataset/simulation/Data_Transformed/Test_MCL/'+subdataset+'/gt_pose.txt')
                                    gt_pose = np.loadtxt(sub_dataset_dir+ '/gt_pose.txt')
                                    particle_evolution = np.load(method_dir+exp+'/particle_set.npy')
                                    odo = np.loadtxt('../Dataset/simulation/Data_Transformed/Test_MCL/'+subdataset+'/odo.txt')
                                    #for o in range(odo.shape[0]):
                                    #    if odo[o,0] == 0 and odo[o,1] == 0 and odo[o,2] == 0:
                                    #        kidnapping_point = o
                                    kidnapping_point = kidnapping_points[dataset+'_'+subdataset]
                                    error_pos = []
                                    error_rot = []

                                    for step in range(particle_evolution.shape[0]-1):
                                        particle_set = particle_evolution[step+1]
                                        w = particle_set[:, 3]
                                        if (np.sum(w)==0):
                                            print(exp)
                                            print(step)
                                        w = w / np.sum(w)
                                        # position error
                                        x_est = np.sum(particle_set[:,0] * w)
                                        y_est = np.sum(particle_set[:,1] * w)
                                        error_pos.append(0.33333*math.sqrt((x_est-gt_pose[step,0])**2+(y_est-gt_pose[step,1])**2))

                                        rots = np.zeros(particle_set.shape[0])
                                        # orientation error
                                        for rot in range(particle_set.shape[0]):
                                            select = np.zeros(3)
                                            select[0] = particle_set[rot,2]
                                            select[1] = particle_set[rot,2] - 2*cfg.pi
                                            select[2] = 2*cfg.pi + particle_set[rot,2]
                                            diff = np.abs(select - gt_pose[step,2])
                                            index = np.argmin(diff)
                                            rots[rot] = select[index]

                                        theta_est = np.sum(rots * w)
                                        error_rot.append(math.sqrt((theta_est-gt_pose[step,2])**2))

                                    error_evolution_pos.append(np.array(error_pos))
                                    error_evolution_rot.append(np.array(error_rot))

                                    st = 0
                                    end = 0
                                    conv_step = 0
                                    while end < kidnapping_point:
                                        if error_pos[end] < T_pos and error_rot[end] < T_rot:
                                            end += 1
                                            if (end - st) > T_step:
                                                conv_step = st
                                                break
                                        else:
                                            end += 1
                                            st = end
                                    if not (conv_step == 0):
                                        conv_step_gl.append(conv_step)
                                    else:
                                        conv_step_gl.append(float('nan'))
                                    error_pos_gl.append(error_pos[kidnapping_point - 2])
                                    error_rot_gl.append(error_rot[kidnapping_point - 2])
                                    '''
                                    if not (conv_step == 0):
                                        # The algorithm converges
                                        avg_error_pos = 0
                                        avg_error_rot = 0
                                        for conv in range(conv_step,conv_step+T_step):
                                            avg_error_pos += error_pos[conv]
                                            avg_error_rot += error_rot[conv]

                                        avg_error_pos /= (T_step)
                                        avg_error_rot /= (T_step)

                                        error_pos_gl.append(avg_error_pos)
                                        error_rot_gl.append(avg_error_rot)
                                        conv_step_gl.append(conv_step)
                                    '''
                                    st = kidnapping_point
                                    end = kidnapping_point
                                    conv_step = kidnapping_point

                                    while end < particle_evolution.shape[0]-1:
                                        if error_pos[end] < T_pos and error_rot[end] < T_rot:
                                            end += 1
                                            if (end - st) > T_step:
                                                conv_step = st
                                                break
                                        else:
                                            end += 1
                                            st = end

                                    if not (conv_step == kidnapping_point):
                                        conv_step_kn.append(conv_step-kidnapping_point)
                                    else:
                                        conv_step_kn.append(float('nan'))
                                    error_pos_kn.append(error_pos[particle_evolution.shape[0] - 2])
                                    error_rot_kn.append(error_rot[particle_evolution.shape[0] - 2])
                                    '''
                                    if not (conv_step == kidnapping_point):
                                        # The algorithm converges in kidnapping problem
                                        avg_error_pos = 0
                                        avg_error_rot = 0

                                        for conv in range(conv_step, conv_step+T_step):
                                            #print(conv)
                                            avg_error_pos += error_pos[conv]
                                            avg_error_rot += error_rot[conv]

                                        avg_error_pos /= (T_step)
                                        avg_error_rot /= (T_step)
                                        conv_step -= kidnapping_point

                                        error_pos_kn.append(avg_error_pos)
                                        error_rot_kn.append(avg_error_rot)
                                        conv_step_kn.append(conv_step)
                                    '''

                            error_pos_gls.append(np.array(error_pos_gl))
                            error_rot_gls.append(np.array(error_rot_gl))
                            step_gls.append(np.array(conv_step_gl))
                            error_pos_kns.append(np.array(error_pos_kn))
                            error_rot_kns.append(np.array(error_rot_kn))
                            step_kns.append(np.array(conv_step_kn))
                    '''
                            R_conv_gl = len(conv_step_gl) / exp_num_gl
                            avg_conv_gl = np.sum(np.array(conv_step_gl)) / len(conv_step_gl)
                            error_pos_gl_mean = np.mean(np.array(error_pos_gl))
                            error_pos_gl_median = np.median(np.array(error_pos_gl))
                            error_pos_gl_var = np.var(np.array(error_pos_gl))
                            error_rot_gl_mean = np.mean(np.array(error_rot_gl))
                            error_rot_gl_var = np.var(np.array(error_rot_gl))
                            error_rot_gl_median = np.median(np.array(error_rot_gl))

                            R_conv_kn = len(conv_step_kn) / exp_num_kn
                            avg_conv_kn = np.sum(np.array(conv_step_kn)) / len(conv_step_kn)
                            error_pos_kn_mean = np.mean(np.array(error_pos_kn))
                            error_pos_kn_median = np.median(np.array(error_pos_kn))
                            error_pos_kn_var = np.var(np.array(error_pos_kn))
                            error_rot_kn_mean = np.mean(np.array(error_rot_kn))
                            error_rot_kn_var = np.var(np.array(error_rot_kn))
                            error_rot_kn_median = np.median(np.array(error_rot_kn))

                            save_results.write('Experiment results for '+method+' in sub-dataset '+subdataset + ' of dataset '+dataset+'\n')
                            save_results.write(str("{:.1f}".format(error_pos_gl_mean))+'&' + str("{:.1f}".format(error_pos_gl_median))+  '&' +str("{:.1f}".format(avg_conv_gl)) +  '&' + str("{:.1f}".format(R_conv_gl*100)) + '&' +'\n')
                            save_results.write(str("{:.1f}".format(error_pos_kn_mean)) +'&' + str("{:.1f}".format(error_pos_kn_median))+ '&' + str("{:.1f}".format(avg_conv_kn)) + '&' + str("{:.1f}".format(R_conv_kn*100)) + '&' + '\n')

                            print('Experiment results for '+method+' in sub-dataset '+subdataset + ' of dataset '+dataset)
                            print(blue('Global Localization: '))
                            print(green('STEPS: ')+str(avg_conv_gl))
                            print(green('Coverging Rate: ')+str(R_conv_gl))
                            print(green('Error in position: ')+str(error_pos_gl_mean)+'+-'+str(error_pos_gl_var))
                            print(green('Error in orientation: ') + str(error_rot_gl_mean) + '+-' + str(error_rot_gl_var))
                            print(blue('Kidnapping: '))
                            print(green('STEPS: ') + str(avg_conv_kn))
                            print(green('Coverging Rate: ') + str(R_conv_kn))
                            print(green('Error in position: ') + str(error_pos_kn_mean) + '+-' + str(error_pos_kn_var))
                            print(green('Error in orientation: ') + str(error_rot_kn_mean) + '+-' + str(error_rot_kn_var))
                    '''
                    '''
                            error_evolution_pos = np.array(error_evolution_pos)
                            error_mean_pos = np.mean(error_evolution_pos,axis=0)
                            error_evolution_rot = np.array(error_evolution_rot)
                            error_mean_rot = np.mean(error_evolution_rot, axis=0)
                            CI_pos = []
                            for error in range(error_evolution_pos.shape[1]):
                                CI_pos.append(sta.norm.interval(0.95, loc=np.mean(error_evolution_pos[:,error]), scale=sta.sem(error_evolution_pos[:,error])))
                            CI_pos = np.array(CI_pos)

                            CI_rot = []
                            for error in range(error_evolution_pos.shape[1]):
                                CI_rot.append(sta.norm.interval(0.95, loc=np.mean(error_evolution_rot[:,error]), scale=sta.sem(error_evolution_rot[:,error])))
                            CI_rot = np.array(CI_rot)

                            methods_error_pos.append(error_mean_pos)
                            methods_error_rot.append(error_mean_rot)
                            methods_CI_pos.append(CI_pos)
                            methods_CI_rot.append(CI_rot)
                    '''
                    '''
                    plt.figure(dpi=500, figsize=(15, 6))
                    for error in range(len(methods_error_pos)):
                        methods_error_pos[error] = np.reshape(methods_error_pos[error],(-1,1))
                    methods_error_pos = np.hstack(methods_error_pos)
                    df = pd.DataFrame(methods_error_pos, columns=methods)
                    f = df.boxplot(sym='o',  # 异常点形状
                                   vert=True,  # 是否垂直
                                   whis=1.5,  # IQR
                                   patch_artist=True,  # 上下四分位框是否填充
                                   meanline=False, showmeans=True,  # 是否有均值线及其形状
                                   showbox=True,  # 是否显示箱线
                                   showfliers=True,  # 是否显示异常值
                                   notch=False,  # 中间箱体是否缺口
                                   return_type='dict')  # 返回类型为字典
                    font = {'size': 14}
                    plt.ylabel('Error in position/m', font)
                    plt.title('箱线图')
                    plt.savefig(sub_dataset_dir + "results.jpg", dpi=500, bbox_inches='tight')
                    '''
                    error_pos_gls = np.array(error_pos_gls)
                    error_rot_gls = np.array(error_rot_gls)
                    step_gls = np.array(step_gls)
                    error_pos_kns = np.array(error_pos_kns)
                    error_rot_kns = np.array(error_rot_kns)
                    step_kns = np.array(step_kns)
                    print(error_pos_gls.shape)
                    print(error_rot_gls.shape)
                    print(step_gls.shape)
                    print(error_pos_kns.shape)
                    print(error_rot_kns.shape)
                    print(step_kns.shape)

                    np.save(sub_dataset_dir + 'error_pos_gls.npy', error_pos_gls)
                    np.save(sub_dataset_dir + 'error_rot_gls.npy', error_rot_gls)
                    np.save(sub_dataset_dir + 'step_gls.npy', step_gls)
                    np.save(sub_dataset_dir + 'error_pos_kns.npy', error_pos_kns)
                    np.save(sub_dataset_dir + 'error_rot_kns.npy', error_rot_kns)
                    np.save(sub_dataset_dir + 'step_kns.npy', step_kns)
                    '''
                    length = len(methods_error_pos)
                    methods_error_pos = np.array(methods_error_pos)
                    methods_error_rot = np.array(methods_error_rot)
                    methods_CI_pos = np.array(methods_CI_pos)
                    methods_CI_rot = np.array(methods_CI_rot)
                    np.save(sub_dataset_dir+'methods_error_pos.npy',methods_error_pos)
                    np.save(sub_dataset_dir + 'methods_CI_pos.npy', methods_CI_pos)
                    x = np.linspace(1,methods_error_pos.shape[1],methods_error_pos.shape[1])
                    l = []
                    plt.figure(dpi=500, figsize=(15, 6))
                    for plot in range(length):
                        plt.plot(x, methods_error_pos[plot], style[plot],label=labels[plot],alpha=0.85)
                        plt.fill_between(x, methods_error_pos[plot], methods_CI_pos[plot,:,0], color=color[plot], alpha=.25)
                        plt.fill_between(x, methods_error_pos[plot], methods_CI_pos[plot, :, 1], color=color[plot], alpha=.25)
                        #dy = np.abs(methods_error_pos[plot]-methods_CI_pos[plot,:,1])
                        #plt.errorbar(x, methods_error_pos[plot], yerr=dy,fmt=error_style[plot],ecolor=error_color[plot],color=error_color[plot],elinewidth=2,capsize=4)

                    plt.title('Converging Process')
                    font = {'size': 14}
                    plt.xlabel('steps',font)
                    plt.ylabel('Error in position/m',font)
                    plt.legend(fontsize=18,loc=1)
                    plt.tick_params(labelsize=14)

                    plt.savefig(sub_dataset_dir+"results.jpg", dpi=500, bbox_inches='tight')
                    '''





