import json, os, time

import torch
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data

from train import load_model, load_from_darknet53

from model.yolo import Darknet
from model.point_net import PointNetBackbone
from model.t_net import PointNetCls 
from model.yolo_pointnet_fuser import FuserPipeline

from trainer.trainer_adaptive_fusion import AdaptiveFusionTrainer

def test_fuser_pipeline(fuser_pipeline:FuserPipeline, dataloader:torch.utils.data.DataLoader):
        
    for batch_idx, data_items in enumerate(dataloader):              
        outputs = fuser_pipeline(
            data_items['images'],
            data_items['raw_point_clouds'],
            data_items['proj2d_pc_mask']
        )
        
        print(outputs[0].shape)
        time.sleep(3)
        
        if batch_idx > 30:
            break
        
def prepare_device_ids(trainer_kwargs:dict):
    yolo_device = torch.device(trainer_kwargs['yolo_device_id']) if torch.cuda.is_available() else torch.device("cpu")        
    pointnet_device = torch.device(trainer_kwargs['pointnet_device_id']) if torch.cuda.is_available() else torch.device("cpu")            
    
    return yolo_device, pointnet_device

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

def create_fuser_pipeline(yolo:Darknet, 
            point_net: PointNetBackbone,
            yolo_device: torch.device,
            point_net_device: torch.device, 
            adaptive_fusion_kwargs: dict):
    
    fuser_pipeline = FuserPipeline(yolo, point_net, 
                                   yolo_device, point_net_device, adaptive_fusion_kwargs)
    
    return fuser_pipeline
    

if __name__ == "__main__":
    
    trainer_config = json.load(open('config/yolo_pointnet_fusion_trainer.json'))
    darknet53_path = ''
    
    # yolo_weights_pth_path = "training_logs/pretrained_darknet53_rgb_Lidar/yolo_weights_59.pth"
    # pointnet_weights_pth_path = '/home/sk4858/CSCI739/model_weights/best_model.pth'
        
    yolo_weights_pth_path = ""
    pointnet_weights_pth_path = ''
        
    yolo_cfg_path = 'config/yolov3-yolo_reduced_classes.cfg'    
    spatial_fusion_ckpt_dir = 'Spatial-Fusion-Pipeline/best-model'
    spatial_fusion_ckpt = 24
    
    yolo_device, pointnet_device = prepare_device_ids(trainer_config['trainer_kwargs'])
    
    if os.path.exists(darknet53_path):
        yolo = load_from_darknet53(yolo_cfg_path,darknet53_path)
    
    else:
        if os.path.exists(yolo_weights_pth_path):
            yolo = load_model(yolo_cfg_path, weights_path=yolo_weights_pth_path)
        else:
            yolo = load_model(yolo_cfg_path)    
    
    yolo.to(yolo_device)
    
    point_net = load_pointnet(trainer_config['pointnet_kwargs'], pointnet_weights_pth_path)
    point_net.to(pointnet_device)
    
    fuser_pipeline = create_fuser_pipeline(yolo, point_net, yolo_device, pointnet_device, trainer_config['adaptive_fusion_kwargs'])
    
    if os.path.exists(spatial_fusion_ckpt_dir):
        fuser_pipeline.load_model_ckpts(spatial_fusion_ckpt_dir, spatial_fusion_ckpt)
    
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