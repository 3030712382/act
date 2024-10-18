import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed
#数据预处理类 继承于torch.utils.data.Dataset 使用 torch.utils.data.Dataset 与 Dataloader 组合得到数据迭代器
class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):#提供数据集大小的方法
        return len(self.episode_ids)

    def __getitem__(self, index):#通过索引号找到数据的方法
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        # print("index:",index)
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        #加载.hdf5文件 文件能够存储和管理非常大的数据集，不受内存限制
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape#该hdf5文件里的动作数量一个任务对应的动作 maxlen
            maxlen_shape = self.norm_stats["maxlen_shape"] 
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]#x y z pitch roll yaw
            qvel = root['/observations/qvel'][start_ts]#vx vy vz 0 0 0 
            targetstate = root['/targetstate'][start_ts]#U yaw
            # print("episode_len: ",episode_len)
            # print("qpos: ",qpos.shape)
            # print("qvel: ",qvel.shape)
            # print("start_ts:",start_ts)
            # print(f"qpos: {qpos}")
            # print(f"qvel: {qvel}")

            image_dict = dict()#加载图像数据
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts 
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned
        # print(action)
        # print("action: ",action.shape)
        self.is_sim = is_sim
        # padded_action = np.zeros(original_action_shape, dtype=np.float32)#展平 
        # print("maxlen_shape:",maxlen_shape)
        padded_action = np.zeros(maxlen_shape, dtype=np.float32)#展平
        padded_action[:action_len] = action
        is_pad = np.zeros(maxlen_shape[0])
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        qvel_data =  torch.from_numpy(qvel).float()#vx vy vz 0 0 0 
        targetstate_data =  torch.from_numpy(targetstate).float()#U yaw
        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float Z-score标准化：将数据转换为均值为0、标准差为1的分布。
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        qvel_data = (qvel_data - self.norm_stats["qvel_mean"]) / self.norm_stats["qvel_std"]
        targetstate_data = (targetstate_data - self.norm_stats["targetstate_mean"]) / self.norm_stats["targetstate_std"]
        qpos_data =  torch.cat((qpos_data, qvel_data))#将速度拼接到位置向量一起进行计算
        # print(qpos_data,qvel_data)
        return image_data, qpos_data, action_data,qvel_data, is_pad


def get_norm_stats(dataset_dir, episodes_range):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in episodes_range:
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')#视频
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]#关节位置
            qvel = root['/observations/qvel'][()]#关节速度
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    # print(all_qpos_data)
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data 标准化动作
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)#均值
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)#方差
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping 限幅

    # normalize qpos data 标准化关节位置
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # 剪裁
    #stats状态
    
    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}
    # print(stats)
    return stats
def calculate_mean_and_std(dataset_dir, episodes_range):
    qpos_sum = None
    action_sum = None
    qvel_sum = None
    targetstate_sum = None
    qvel_count = 0
    targetstate_count = 0
    qpos_count = 0
    action_count = 0
    maxlen_shape = (0,0)
    # 第一遍：计算总和和元素数量
    for episode_idx in episodes_range:
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
            qvel = root['/observations/qvel'][()]
            targetstate = root['/targetstate'][()]
            if(action.shape[0]>=maxlen_shape[0]):
                maxlen_shape = action.shape
        # 初始化总和
        if qpos_sum is None:
            qpos_sum = np.zeros_like(qpos[0])
            action_sum = np.zeros_like(action[0])
            qvel_sum = np.zeros_like(qvel[0])
            targetstate_sum = np.zeros_like(targetstate[0])

        # 更新总和和元素数量
        qpos_sum += np.sum(qpos, axis=0)
        action_sum += np.sum(action, axis=0)
        qvel_sum += np.sum(qvel, axis=0)
        targetstate_sum += np.sum(targetstate, axis=0)
        qpos_count += qpos.shape[0]
        action_count += action.shape[0]
        qvel_count += qvel.shape[0]
        targetstate_count += targetstate.shape[0]

    # 计算均值
    qpos_mean = qpos_sum / qpos_count
    action_mean = action_sum / action_count
    qvel_mean = qvel_sum / qvel_count
    targetstate_mean = targetstate_sum / targetstate_count

    # 第二遍：计算方差
    qpos_sq_diff_sum = np.zeros_like(qpos_mean)
    action_sq_diff_sum = np.zeros_like(action_mean)
    targetstate_sq_diff_sum = np.zeros_like(targetstate_mean)
    qvel_sq_diff_sum = np.zeros_like(qvel_mean)

    for episode_idx in episodes_range:
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
            qvel = root['/observations/qvel'][()]
            targetstate = root['/targetstate'][()]
        
        # 累加差值的平方
        qpos_sq_diff_sum += np.sum((qpos - qpos_mean) ** 2, axis=0)
        action_sq_diff_sum += np.sum((action - action_mean) ** 2, axis=0)
        qvel_sq_diff_sum += np.sum((qvel - qvel_mean) ** 2,axis=0)
        targetstate_sq_diff_sum += np.sum((targetstate - targetstate_mean) ** 2, axis=0)

    # 计算标准差
    qpos_std = np.sqrt(qpos_sq_diff_sum / qpos_count)
    action_std = np.sqrt(action_sq_diff_sum / action_count)
    qvel_std = np.sqrt(qvel_sq_diff_sum / qvel_count)
    qvel_std[3:] = 1#角速度没有用上 原始数据为0 置为1 避免除法nan
    targetstate_std = np.sqrt(targetstate_sq_diff_sum / targetstate_count)
    # print(qvel_std,targetstate_std,qpos_std,action_std)
    return {"qpos_mean": qpos_mean, "qpos_std": qpos_std,
            "action_mean": action_mean, "action_std": action_std,"maxlen_shape":maxlen_shape,
            "qvel_mean": qvel_mean, "qvel_std": qvel_std,
            "targetstate_mean": targetstate_mean, "targetstate_std": targetstate_std}

#数据加载  num_episodes：50次
def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    files = os.listdir(dataset_dir)
    num_episodes = len([f for f in files if os.path.join(dataset_dir, f).endswith('.hdf5')])
    print("num_episodes: ", num_episodes)
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split 数据划分
    train_ratio = 0.8
    # shuffled_indices = np.random.permutation(num_episodes)#打乱顺序 np.arange(1, n + 1)
    shuffled_indices = np.random.permutation(np.arange(1, num_episodes + 1 ))#np.arange(1, num_episodes + 1)
    print(shuffled_indices)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]#40
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]#10

    # obtain normalization stats for qpos and action
    print(dataset_dir)
    norm_stats = calculate_mean_and_std(dataset_dir, shuffled_indices)#计算数据的均值与方差 get_norm_stats
 
    # construct dataset and dataloader加载数据并预处理
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    #数据集批次化
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
if __name__ == '__main__':
    # EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    dataset_dir = '/home/hp-t4/data/berthing' # sim_insertion_human
    num_episodes = 50
    camera_names = ['front_left_camera', 'front_right_camera']
    batch_size_train = 4
    batch_size_val = 1
    train_dataloader, val_dataloader, stats, _  = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)
    for batch_idx, data in enumerate(train_dataloader):
        # print(batch_idx,data)
        print(batch_idx)
