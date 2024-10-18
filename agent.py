import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import rospy,sys
from .constants import DT
from .constants import PUPPET_GRIPPER_JOINT_OPEN
from .utils import load_data # data functions
from .utils import sample_box_pose, sample_insertion_pose # robot functions
from .utils import compute_dict_mean, set_seed, detach_dict # helper functions
from .policy import ACTPolicy, CNNMLPPolicy
# from .sim_env import BOX_POSE

import IPython
e = IPython.embed
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', default="/home/hp-t4/imlearning/sim_berthing",required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize',default="ACT", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default="sim_berthing",required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=4,required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0,required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs',default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr',default=1e-5, required=False)
    parser.add_argument('--state_dim', action='store', type=float, help='state_dim', default=12,required=False)
    parser.add_argument('--action_dim', action='store', type=float, help='action_dim', default=2,required=False)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10,required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', default=100,required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512,required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward',default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    return parser
def parse_arguments():
    # 如果是通过 ROS 运行
    if rospy.get_param_names():
        args = {}
        args['task_name'] = rospy.get_param('~task_name', 'default_task')
        args['ckpt_dir'] = rospy.get_param('~ckpt_dir', '/default/path')
        args['policy_class'] = rospy.get_param('~policy_class', 'default_policy')
        args['kl_weight'] = rospy.get_param('~kl_weight', 10)
        args['chunk_size'] = rospy.get_param('~chunk_size', 100)
        args['hidden_dim'] = rospy.get_param('~hidden_dim', 512)
        args['batch_size'] = rospy.get_param('~batch_size', 4)
        args['dim_feedforward'] = rospy.get_param('~dim_feedforward', 3200)
        args['num_epochs'] = rospy.get_param('~num_epochs', 2000)
        args['lr'] = rospy.get_param('~lr', 1e-5)
        args['state_dim'] = rospy.get_param('~state_dim', 12)
        args['action_dim'] = rospy.get_param('~action_dim', 2)
        args['seed'] = rospy.get_param('~seed', 0)
        args['eval'] = rospy.get_param('~eval',False)
        args['onscreen_render'] = rospy.get_param('~onscreen_render',False)
        args['temporal_agg'] = rospy.get_param('~temporal_agg',False)
        # 将参数转为命令行格式
        return argparse.Namespace(**args)
    
    # 否则通过命令行运行
    else:
        parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
        return parser.parse_args()


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

class professor:
    def __init__(self,args):
        set_seed(1)
        # command line parameters
        self.is_eval = args['eval']
        self.ckpt_dir = args['ckpt_dir']#模型生成地址
        self.policy_class = args['policy_class']
        self.onscreen_render = args['onscreen_render']
        self.task_name = args['task_name']
        self.batch_size_train = args['batch_size']
        self.batch_size_val = args['batch_size']#4
        self.num_epochs = args['num_epochs']#num_steps/batch_size
        # get task parameters
        is_sim = self.task_name[:4] == 'sim_'
        if is_sim:
            from .constants import SIM_TASK_CONFIGS
            task_config = SIM_TASK_CONFIGS[self.task_name]
        # else:
        #     from aloha_scripts.constants import TASK_CONFIGS
        #     task_config = TASK_CONFIGS[task_name]
        self.dataset_dir = task_config['dataset_dir']
        self.num_episodes = task_config['num_episodes']
        self.episode_len = task_config['episode_len']
        self.camera_names = task_config['camera_names']

        # fixed parameters
        self.state_dim = args['state_dim']#机器人状态向量的大小12
        self.action_dim = args['action_dim']
        self.lr_backbone = 1e-5
        self.backbone = 'resnet18'
        if self.policy_class == 'ACT':
            enc_layers = 4 #encode 编码层数
            dec_layers = 7
            nheads = 8
            self.policy_config = {'lr': args['lr'],
                            'num_queries': args['chunk_size'],#100
                            'kl_weight': args['kl_weight'],#10 kl权重
                            'hidden_dim': args['hidden_dim'],#隐藏层的大小深度 512 即图片特征参数
                            'dim_feedforward': args['dim_feedforward'],#3200
                            'lr_backbone': self.lr_backbone,
                            'backbone': self.backbone,
                            'enc_layers': enc_layers,
                            'dec_layers': dec_layers,
                            'nheads': nheads,
                            'camera_names': self.camera_names,
                            }
        elif self.policy_class == 'CNNMLP':
            self.policy_config = {'lr': args['lr'], 'lr_backbone': self.lr_backbone, 'backbone' : self.backbone, 'num_queries': 1,
                            'camera_names': self.camera_names,}
        else:
            raise NotImplementedError
        #
        self.config = {
            'num_epochs': self.num_epochs,
            'ckpt_dir': self.ckpt_dir,
            'episode_len': self.episode_len,
            'state_dim': self.state_dim,
            'lr': args['lr'],
            'policy_class': self.policy_class,
            'onscreen_render': self.onscreen_render,
            'policy_config': self.policy_config,
            'task_name': self.task_name,
            'seed': args['seed'],
            'temporal_agg': args['temporal_agg'],
            'camera_names': self.camera_names,
            'real_robot': not is_sim
        }
        set_seed(1000)
        ckpt_path = os.path.join(self.ckpt_dir, f'policy_last.ckpt')
        self.policy = make_policy(self.policy_class, self.policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(self.ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        self.t = 0
        ### evaluation loop
        self.pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        self.pre_processq_vel = lambda s_qvel: (s_qvel - self.stats['qvel_mean']) / self.stats['qvel_std']
        self.post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
        if self.policy_class == 'Diffusion':
            self.post_process = lambda a: ((a + 1) / 2) * (self.stats['action_max'] - self.stats['action_min']) + self.stats['action_min']
        else:
            self.post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']

        self.query_frequency = self.policy_config['num_queries']
        if self.config["temporal_agg"]:
            self.query_frequency = 1
            self.num_queries = self.policy_config['num_queries']
        self.max_timesteps = int(self.episode_len * 1) # may increase for real-world tasks
        if self.config["temporal_agg"]:
            self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, self.action_dim]).cuda()
        self.qpos_history = torch.zeros((1, self.max_timesteps, self.state_dim)).cuda()
        self.image_list = [] # for visualization
        self.qpos_list = []
        self.target_qpos_list = []
        self.rewards = []
        self.num_rollouts = 50
        self.episode_returns = []
        self.highest_rewards = []
        self.t = 0
    def infer(self,cameras,qpos,qvel):
        qvel_data =  torch.from_numpy(qvel).float()#vx vy vz 0 0 0 
        qpos_data = torch.from_numpy(qpos).float()
        qpos_data = self.pre_process(qpos_data)
        qvel_data = self.pre_processq_vel(qvel_data)
        qpos  =  torch.cat((qpos_data, qvel_data)).float().cuda().unsqueeze(0)#将速度拼接到位置向量一起进行计算
        with torch.inference_mode():
            if self.t <self.max_timesteps-1:
                
                ### process previous timestep to get qpos and image_list
                # qpos = self.pre_process(qpos_numpy)
                # qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                self.qpos_history[:, self.t] = qpos.cuda()
                #获取相机图片
                all_cam_images = []
                for cam_name in self.camera_names:
                    all_cam_images.append(cameras[cam_name])
                all_cam_images = np.stack(all_cam_images, axis=0)
                image_data = torch.from_numpy(all_cam_images)
                image_data = torch.einsum('k h w c -> k c h w', image_data)
                curr_image = image_data / 255.0
                curr_image = curr_image.cuda().unsqueeze(0).repeat(1, 1, 1, 1, 1)
                ### query policy
                if self.config['policy_class'] == "ACT":
                    # if self.t % self.query_frequency == 0:
                    all_actions = self.policy(qpos, curr_image.cuda())
                    if self.config["temporal_agg"]:
                        self.all_time_actions[[self.t], self.t:self.t+self.num_queries] = all_actions[0]
                        actions_for_curr_step = self.all_time_actions[:, self.t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, self.t % self.query_frequency]
                elif self.config['policy_class'] == "CNNMLP":
                    raw_action = self.policy(qpos, curr_image)
                else:
                    raise NotImplementedError
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = self.post_process(raw_action)
                self.target_qpos = action
                self.t += 1
                ### for visualization
                self.qpos_list.append(qpos)
                self.target_qpos_list.append(self.target_qpos)
            else:
                self.t = 0
            return self.target_qpos
            


    def train(self):
        #数据加载
        train_dataloader, val_dataloader, stats, _ = load_data(self.dataset_dir, self.num_episodes, self.camera_names, self.batch_size_train, self.batch_size_val)

        # save dataset stats
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        stats_path = os.path.join(self.ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        #训练

if __name__ == '__main__':

        
    
    boat = professor(vars(parser.parse_args()))

    # boat.infer()
