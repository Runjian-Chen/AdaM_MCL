import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append("..")
import argparse
from config.config import cfg
from src.train import train
from src.Global_Localization_Interface import Global_Localization_Interface
from src.post_process_logs import post_process_logs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
        Arguments for selecting operation modes among training, testing for the network, testing for MCL and data generation.
    """
    parser.add_argument("--operation", help="Select operation among train, test_MCL, post_process_logs", default='train')

    """
        Arguments for universal parameters.
    """
    parser.add_argument("--dataset", help="Select dataset among Simulation, Bicocca and UPO", default='Bicocca')
    parser.add_argument("--subdataset", help="Select sub-dataset among 1,2,...", default='1')
    parser.add_argument("--load_model", help="model path to load", default=None)
    parser.add_argument("--device", help="device used", default=cfg.DSOM_device)

    """
        Arguments for parameters required in training.
    """
    parser.add_argument("--num_epochs", help="the number of epochs to train", default=cfg.DSOM_num_epoch)
    parser.add_argument("--batchsize", help="the number of epochs to train", default=cfg.DSOM_batchsize)
    parser.add_argument("--num_iters_per_epoch_train", help="the number of iteration in each epoch when training", default=cfg.DSOM_training_iters)
    parser.add_argument("--num_iters_per_epoch_val", help="the number of iteration in validation", default=cfg.DSOM_validation_iters)
    parser.add_argument("--save_per_x_epoch", help="save model per x epoch", default=2)
    parser.add_argument("--decay_per_x_step", help="decay per x global steps", default=40000)
    parser.add_argument("--decay_rate", help="decay rate", default=0.999)
    parser.add_argument("--val_per_x_iters", help="validate per x iterations", default=3600)

    """
        Arguments for parameters required in testing for the network.
    """
    parser.add_argument("--num_iters_per_epoch_testing", help="the number of iteration in testing", default=cfg.DSOM_validation_iters)

    """
        Arguments for parameters required in testing for the MCL.
    """
    parser.add_argument("--particle_num", help="The number of particles used in the filter", default=1000)
    parser.add_argument("--update_mode", help="The mode used in the filter", default='DL_Aided_Particle_Filter')
    parser.add_argument("--random_sampling", help="Random sample or not", default=True)
    parser.add_argument("--random_sample_rate", help="Random sample rate", default=0.2)
    parser.add_argument("--w_cut", help="parameter for cutting weight", default=0.6)
    parser.add_argument("--rc_lamda_short", help="lamda for short model", default=1)
    parser.add_argument("--rc_w_hit", help="Weight for hit model", default=0.95)
    parser.add_argument("--rc_w_short", help="Weight for short model", default=1)
    parser.add_argument("--rc_w_max", help="Weight for max model", default=1)
    parser.add_argument("--rc_w_rand", help="Weight for random model", default=0.05)
    parser.add_argument("--p_mcl", help="probability to execute MCL in Mixture MCL", default=0.95)
    args = parser.parse_args()

    if args.dataset == "simulation":
        cfg.DSOM_X = 129
        cfg.DSOM_Y = 129
        cfg.DSOM_batchsize = 32
        args.batchsize = 32
    else:
        cfg.DSOM_X = 320
        cfg.DSOM_Y = 320
        cfg.DSOM_batchsize = 16
        args.batchsize = 16

    if args.operation == "train":
        train_model = train(args.dataset,args.subdataset,int(args.batchsize), int(args.num_iters_per_epoch_train), args.num_iters_per_epoch_val, args.num_epochs, args.device, args.load_model, args.save_per_x_epoch, int(args.decay_per_x_step), args.decay_rate, args.val_per_x_iters)
        train_model.train()

    elif args.operation == "test_MCL":
        cfg.DSOM_batchsize = 1
        args.batchsize = 1
        cfg.DSOM_model_path = args.load_model
        if args.dataset == 'simulation':
            paras = {'Random_Produce': args.random_sampling, 'scan_res': cfg.simulation_Dataset_angle_step,
                     'scan_angle_st': cfg.simulation_Dataset_st_angle,
                     'max_scan_range': cfg.simulation_Dataset_max_range,
                     'var_rc_weighting_model': 5,
                     'laser_portions': cfg.simulation_Dataset_scan_steps, 'Random_sample_var_x': 0.1, 'Random_sample_var_y': 0.1,
                     'Random_sample_var_theta': 0.1, 'w_cut': float(args.w_cut),
                     'rc_lamda_short': float(args.rc_lamda_short),
                     'rc_w_hit': float(args.rc_w_hit), 'rc_w_short': float(args.rc_w_short),
                     'rc_w_max': float(args.rc_w_max), 'rc_w_rand': float(args.rc_w_rand),
                     'device': 'cpu', 'laser_steps': 5, 'random_sample_rate': args.random_sample_rate,'P_MCL': float(args.p_mcl),
                     'resolution':cfg.simulation_Dataset_resolution,
                     }
        elif args.dataset == 'Bicocca':
            paras = {'Random_Produce': args.random_sampling, 'scan_res': cfg.Bicocca_angle_step,
                     'scan_angle_st': cfg.Bicocca_st_angle,
                     'max_scan_range': cfg.Bicocca_laser_max_range,
                     'var_rc_weighting_model': 5,
                     'laser_portions': cfg.Bicocca_scan_steps, 'Random_sample_var_x': 0.1,
                     'Random_sample_var_y': 0.1,
                     'Random_sample_var_theta': 0.1, 'w_cut': float(args.w_cut),
                     'rc_lamda_short': float(args.rc_lamda_short),
                     'rc_w_hit': float(args.rc_w_hit), 'rc_w_short': float(args.rc_w_short),
                     'rc_w_max': float(args.rc_w_max), 'rc_w_rand': float(args.rc_w_rand),
                     'device': 'cpu', 'laser_steps': 5, 'random_sample_rate': args.random_sample_rate,
                     'P_MCL': float(args.p_mcl),
                     'resolution': cfg.Bicocca_resolution,
                     }

        elif args.dataset == 'UPO':
            paras = {'Random_Produce': args.random_sampling, 'scan_res': cfg.UPO_Dataset_angle_increment,
                     'scan_angle_st': cfg.UPO_Dataset_angle_min,
                     'max_scan_range': cfg.UPO_Dataset_range_max,
                     'var_rc_weighting_model': 5,
                     'laser_portions': cfg.UPO_Dataset_angle_portions, 'Random_sample_var_x': 0.1,
                     'Random_sample_var_y': 0.1,
                     'Random_sample_var_theta': 0.1, 'w_cut': float(args.w_cut),
                     'rc_lamda_short': float(args.rc_lamda_short),
                     'rc_w_hit': float(args.rc_w_hit), 'rc_w_short': float(args.rc_w_short),
                     'rc_w_max': float(args.rc_w_max), 'rc_w_rand': float(args.rc_w_rand),
                     'device': 'cpu', 'laser_steps': 5, 'random_sample_rate': args.random_sample_rate,
                     'P_MCL': float(args.p_mcl),
                     'resolution': cfg.UPO_Dataset_resolution,
                     }
        for exp in range(100):
            Interface = Global_Localization_Interface(args.dataset,args.subdataset,args.update_mode,args.particle_num,paras)

            Interface.update_pf()
    elif args.operation == "post_process_logs":
        post_process_logs('./logs/',1,10 * 3.14159 / 180,10)
    else:
        assert ('No operation selected!')