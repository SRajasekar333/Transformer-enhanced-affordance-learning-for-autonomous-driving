import argparse
import os
import datetime

from tqdm import tqdm
import numpy as np
import json
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from config1 import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data

### full data set without town2 nad town 5 ###

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default=r'/data/horse/ws/rasa397c-Sankar/transfuser/data_mini', help='Root directory of your training data')
parser.add_argument('--model_path', type=str, required=True, help='path to model ckpt')
parser.add_argument('--args_path', type=str, required=True, help='path to training arguments')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--save_path', type=str, default=None, help='path to save visualizations')
parser.add_argument('--total_size', type=int, default=100000000, help='total images for which to generate visualizations')
parser.add_argument('--attn_thres', type=int, default=1, help='minimum # tokens of other modality required for global context')
# parser.add_argument('--backbone', type=str, default='transFuser', help='Which vision backbone to use. Options: geometric_fusion, transFuser, late_fusion, aim')
parser.add_argument('--setting', type=str, default='viz', help='What training setting to use. Options: '
                                                                   'all: Train on all towns no validation data. '
                                                                   '02_05_withheld: Do not train on Town 02 and Town 05. Use the data as validation data.')

args = parser.parse_args()

with open(args.args_path, 'r') as f:
	training_args = json.load(f)

# Config
#config = GlobalConfig()
config = GlobalConfig(root_dir=args.root_dir, setting=args.setting)
config.use_target_point_image = bool(training_args['use_target_point_image'])
config.n_layer = training_args['n_layer']

if args.save_path is not None and not os.path.isdir(args.save_path):
	os.makedirs(args.save_path, exist_ok=True)

# Data
viz_data = CARLA_Data(root=config.viz_data, config=config)
#viz_data = CARLA_Data(root=config.train_data, config=config)
dataloader_viz = DataLoader(viz_data, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=False)

# Model
model = LidarCenterNet(config, args.device, training_args['backbone'], training_args['image_architecture'],training_args['lidar_architecture'], bool(training_args['use_velocity']))
model = torch.nn.DataParallel(model) # change to distributed data parallel later

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('Total parameters: ', params)

state_dict = torch.load(os.path.join(args.model_path), map_location=args.device)
model.load_state_dict(state_dict, strict=False)
model.cuda()
model.eval()

total_images_processed = 0

with torch.no_grad():
    for enum, data in enumerate(tqdm(dataloader_viz)):
        if enum*args.batch_size >= args.total_size:
            break
        rgb = data['rgb'].to(args.device, dtype=torch.float32)
        #print(rgb.size)
        lidar = data['lidar'].to(args.device, dtype=torch.float32)
        target_point_image = data['target_point_image'].to(args.device, dtype=torch.float32)
        ego_vel = data['speed'].to(args.device, dtype=torch.float32).reshape(-1, 1)

        # get geometric projections to visualize geometric fusion
        #bev_points = data['bev_points'][-1].numpy().astype(np.uint8)
        #cam_points = data['cam_points'][-1].numpy().astype(np.uint8)
        #bs, _, _, n_corres, p_dim = bev_points.shape
        #bev_points = bev_points.reshape((bs, -1, n_corres, p_dim))
        #cam_points = cam_points.reshape((bs, -1, n_corres, p_dim))
        #bev_points = data['bev_points'][0].long().to(args.device, dtype=torch.int64)
        #cam_points = data['cam_points'][0].long().to(args.device, dtype=torch.int64)

        
        if config.use_target_point_image:
            lidar = torch.cat((lidar, target_point_image), dim=1)
        _, _, fused_features = model.module._model(rgb, lidar, ego_vel)
        #_, _, fused_features = model.module._model(rgb, lidar, ego_vel, bev_points, cam_points)
        for idx in range(fused_features.size(0)):
            feature_path = os.path.join(args.save_path, f'batch_{enum}_idx_{idx}.pt')
            torch.save(fused_features[idx].cpu(), feature_path)
            #print(f'batch {enum}, Sample {idx}, Fused Feature: {fused_features[idx]}')
        total_images_processed += fused_features.size(0)

print(f'Total images processed: {total_images_processed}')


