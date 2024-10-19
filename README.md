# act
本项目首次将模仿学习算法应用于船舶领域
The imitation learning code for berthing
模仿学习代码 参考项目：https://github.com/MarkFzp/act-plus-plus
## Repo Structure
- ``imitate_episodes.py`` Train  ACT
- ``agent.py`` Test or Run ACT 
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


## Installation
这里只给出如何单独下载代码并安装，若想要结合船舶GAZEBO仿真或者部署实验请看项目：
Project Website: https://github.com/3030712382/haitun2/tree/v1.0

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .

- also need to install https://github.com/ARISE-Initiative/robomimic/tree/r2d2 (note the r2d2 branch) for Diffusion Policy by `pip install -e .`

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments (LEGACY table-top ALOHA environments)

We use ``sim_berthing`` task in the examples below. 
To generated 50 episodes of scripted data one by one, run:
 see the instruction in act, https://github.com/ARISE-Initiative/robomimic/tree/r2d2

To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py --task_name sim_berthing --ckpt_dir /home/hp-t4/imlearning/sim_berthing --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 4 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0


