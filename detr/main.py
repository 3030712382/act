# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model
import rospy,sys
import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=False)#True
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=False)#True
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=False)#True
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False)#True
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=False)#True
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
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
        args['seed'] = rospy.get_param('~seed', 0)
        args['eval'] = rospy.get_param('~eval',False)
        args['lr_backbone'] = rospy.get_param('~lr_backbone', 1e-5)
        args['weight_decay'] = rospy.get_param('~weight_decay', 1e-5)
        args['epochs'] = rospy.get_param('~epochs', 300)
        args['lr_drop'] = rospy.get_param('~lr_drop', 200)# not used
        args['clip_max_norm'] = rospy.get_param('~clip_max_norm', 0.1)# not used


        # Model parameters
         # * Backbone
        args['backbone'] = rospy.get_param('~backbone',"resnet18")
        args['dilation'] = rospy.get_param('~dilation',False)
        args['position_embedding'] = rospy.get_param('~position_embedding',"sine")# choices=('sine', 'learned')
        args['camera_names'] = rospy.get_param('~camera_names',[])# choices=('sine', 'learned')
        # * Transformer
        args['enc_layers'] = rospy.get_param('~enc_layers', 4)
        args['dec_layers'] = rospy.get_param('~dec_layers', 6)
        args['dim_feedforward'] = rospy.get_param('~dim_feedforward', 2048)
        args['hidden_dim'] = rospy.get_param('~hidden_dim', 256)
        args['dropout'] = rospy.get_param('~dropout', 0.1)
        args['nheads'] = rospy.get_param('~nheads', 8)
        args['dropout'] = rospy.get_param('~dropout', 0.1)
        args['num_queries'] = rospy.get_param('~num_queries',400)
        args['pre_norm'] = rospy.get_param('~pre_norm',False)
        # * Segmentation
        args['masks'] = rospy.get_param('~masks',False)
        # 将参数转为命令行格式
        return argparse.Namespace(**args)
    
    # 否则通过命令行运行
    else:
        parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
        return parser.parse_args()


def build_ACT_model_and_optimizer(args_override):
    # parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parse_arguments()
    for k, v in args_override.items():
        setattr(args, k, v)
    print("dec_layers: ",args.dec_layers)
    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

def build_CNNMLP_model_and_optimizer(args_override):
    # parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parse_arguments()
    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

