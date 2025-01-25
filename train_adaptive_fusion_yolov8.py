import json, os, time

import torch
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from ultralytics import YOLO

from model.point_net import PointNetBackbone
from model.t_net import PointNetCls 
from model.yolov8_pointnet_fuser import Yolov8FuserPipeline

from trainer.trainer_adaptive_fusion_yolov8 import AdaptiveFusionTrainer


        
def prepare_device_ids(trainer_kwargs:dict):
    yolo_device = torch.device(trainer_kwargs['yolo_device_id']) if torch.cuda.is_available() else torch.device("cpu")        
    pointnet_device = torch.device(trainer_kwargs['pointnet_device_id']) if torch.cuda.is_available() else torch.device("cpu")            
    
    return yolo_device, pointnet_device

def load_yolov8(yaml_path:str, weights_path:str):
    yolo_v8n = YOLO(yaml_path)
    
    if os.path.exists(weights_path):
        print(f'YOLOv8 Model loaded')
        yolo_v8n.load(
            weights_path
        ) 
        
    return yolo_v8n   

def load_pointnet(pointnet_kwargs:dict, weights_path:str):
    
    # point_net = PointNetBackbone(
    #     num_points=pointnet_kwargs['num_points'],
    #     num_global_feats=pointnet_kwargs['num_global_feats']
    # )    
    
    point_net = PointNetCls(num_classes=9)
    
    if os.path.exists(weights_path):
        point_net.load_state_dict(
            torch.load(weights_path)['model_state_dict']
        )

    return point_net.feature_transform

def create_fuser_pipeline(yolo_v8n:YOLO, 
            point_net: PointNetBackbone,
            yolo_device: torch.device,
            point_net_device: torch.device, 
            adaptive_fusion_kwargs: dict, 
            image_resize):
    
    fuser_pipeline = Yolov8FuserPipeline(yolo_v8n, point_net, 
        yolo_device, point_net_device,
        adaptive_fusion_kwargs, 
        image_resize)
    
    return fuser_pipeline
    

if __name__ == "__main__":
    
    trainer_config = json.load(open('config/yolov8_pointnet_fusion_trainer.json'))
    darknet53_path = ''
    
    yolov8_weights_pth_path = "yolov8n.pt"
    yolov8_yaml_path = 'config/yolov8-KiTTi.yaml'
    pointnet_weights_pth_path = '/home/sk4858/CSCI739/model_weights/best_model.pth'        
            
    spatial_fusion_ckpt_dir = ''
    spatial_fusion_ckpt = 0
    
    yolo_device, pointnet_device = prepare_device_ids(trainer_config['trainer_kwargs'])

    yolo = load_yolov8(yolov8_yaml_path, yolov8_weights_pth_path) 
    yolo.to(yolo_device)
    
    point_net = load_pointnet(trainer_config['pointnet_kwargs'], pointnet_weights_pth_path)
    point_net.to(pointnet_device)
    
    fuser_pipeline = create_fuser_pipeline(
        yolo, point_net, 
        yolo_device, pointnet_device, 
        trainer_config['adaptive_fusion_kwargs'], 
        trainer_config['dataset_kwargs']['image_resize'])
    
    if os.path.exists(spatial_fusion_ckpt_dir):
        '''TODO'''
        pass
    
    trainer = AdaptiveFusionTrainer(
        fuser_pipeline,  
        trainer_config['dataset_kwargs'],
        trainer_config['optimizer_kwargs'],
        trainer_config['trainer_kwargs'],
        trainer_config['lr_scheduler_kwargs']        
    )    
    trainer.train()
    # test_fuser_pipeline(
    #     fuser_pipeline, 
    #     trainer.train_dataloader
    # )