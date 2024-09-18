import argparse
import os
import datetime
import csv

from tqdm import tqdm
import numpy as np
import json
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from config1 import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default=r'/data/horse/ws/rasa397c-Sankar/transfuser/data_mini', help='Root directory of your training data')
parser.add_argument('--model_path', type=str, required=True, help='path to model ckpt')
parser.add_argument('--args_path', type=str, required=True, help='path to training arguments')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--save_path', type=str, default=None, help='path to save visualizations')
parser.add_argument('--total_size', type=int, default=1000000, help='total images for which to generate visualizations')
parser.add_argument('--attn_thres', type=int, default=1, help='minimum # tokens of other modality required for global context')
# parser.add_argument('--backbone', type=str, default='transFuser', help='Which vision backbone to use. Options: geometric_fusion, transFuser, late_fusion, aim')
parser.add_argument('--setting', type=str, default='viz', help='What training setting to use. Options: '
                                                                   'all: Train on all towns no validation data. '
                                                                   '02_05_withheld: Do not train on Town 02 and Town 05. Use the data as validation data.')

args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

csv_file_path = os.path.join(args.save_path, 'annotations_trans_preddata.csv')

# Function to process hazard lists and determine the majority value
def get_majority_hazard(hazard_list):
    true_count = sum(hazard_list)
    false_count = len(hazard_list) - true_count
    return "true" if true_count > false_count else "false"

# Config
config = GlobalConfig(root_dir=args.root_dir, setting=args.setting)

# Data
#viz_data = CARLA_Data(root=config.train_data, config=config)
viz_data = CARLA_Data(root=config.viz_data, config=config)
#dataloader_viz = DataLoader(viz_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

from torch.utils.data.dataloader import default_collate

##############   fullloader check   ###############
def custom_collate_fn(batch):
    # Filter out NoneType items from the batch
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    return default_collate(batch)

from torch.utils.data import DataLoader

# Use the custom collate function in your DataLoader
dataloader_viz = DataLoader(viz_data, batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)
##############   fullloader check   ###############

total_junction_count = 0

# Open CSV file to write
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write CSV header
    csv_writer.writerow(['batch', 'idx', 'target_speed', 'brake', 'junction', 'vehicle_hazard', 'light_hazard', 'walker_hazard', 'stop_sign_hazard', 'rel_angle', 'lateral_distance', 'distance'])

    for enum, data in enumerate(tqdm(dataloader_viz)):
        #if data is None:  
            #print(f"Batch {enum} is empty.")
            #continue
        target_speed = data['target_speed'].to('cpu', dtype=torch.float32)
        brake = data['brake'].to('cpu', dtype=torch.float32)
        junction = data['junction'].to('cpu', dtype=torch.float32)
        vehicle_hazard = data['vehicle_hazard']
        light = data['light'].to('cpu', dtype=torch.float32)
        walker_hazard = data['walker_hazard']
        stop_sign_hazard = data['stop_sign_hazard'].to('cpu', dtype=torch.float32)
        rel_angle = data['angle'].to('cpu', dtype=torch.float32)
        lateral_distance = data['mean_lateral_distance'].to('cpu', dtype=torch.float32)
        distance = data['distance'].to('cpu', dtype=torch.float32)

        for idx in range(light.size(0)):
            brake_str = "true" if bool(brake[idx].item()) else "false"
            junction_str = "true" if bool(junction[idx].item()) else "false"
            light_str = "true" if bool(light[idx].item()) else "false"
            stop_sign_hazard_str = "true" if bool(stop_sign_hazard[idx].item()) else "false"

            # Process vehicle_hazard and walker_hazard
            vehicle_hazard_str = "false"
            walker_hazard_str = "false"
            if len(vehicle_hazard) > idx:
                vehicle_hazard_true_count = sum(vehicle_hazard[idx])
                vehicle_hazard_false_count = len(vehicle_hazard[idx]) - vehicle_hazard_true_count
                vehicle_hazard_str = "true" if vehicle_hazard_true_count > vehicle_hazard_false_count else "false"

            if len(walker_hazard) > idx:
                walker_hazard_true_count = sum(walker_hazard[idx])
                walker_hazard_false_count = len(walker_hazard[idx]) - walker_hazard_true_count
                walker_hazard_str = "true" if walker_hazard_true_count > walker_hazard_false_count else "false"

            csv_writer.writerow([
                enum, 
                idx, 
                target_speed[idx].item(), 
                brake_str, 
                junction_str, 
                vehicle_hazard_str, 
                light_str, 
                walker_hazard_str, 
                stop_sign_hazard_str, 
                rel_angle[idx].item(), 
                lateral_distance[idx].item(), 
                distance[idx].item()
            ])

        total_junction_count += junction.size(0)

print(f'Total count of junction data: {total_junction_count}')
print(f'Data saved to CSV file: {csv_file_path}')
