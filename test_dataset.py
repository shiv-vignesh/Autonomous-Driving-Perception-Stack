import torch, time, os
from torch.utils.data import DataLoader

from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiLidarFusionCollateFn
from trainer.loss import compute_loss
# from model.yolo import Darknet
from ultralytics import YOLO
from model.yolov8_pointnet_fuser import Yolov8FuserPipeline
from model.point_net import PointNetBackbone
from model.t_net import PointNetCls 

if __name__ == "__main__":
    
    # yolo = Darknet(
    #     'config/yolov3-KiTTi.cfg'
    # )
    
    pointnet_weights_path = '/home/sk4858/CSCI739/model_weights/best_model.pth'
    yolov8_weights_path = 'yolov8n.pt'
    
    adaptive_fusion_kwargs = {
        "transform_image_features":False, 
        "fusion_type":"residual",
        "alpha":1.0
    }    
    image_resize=(640, 640)
    grid_sizes=[(20, 20), (40, 40), (80, 80)]
    yolo_device = torch.device('cuda:2')
    pointnet_device = torch.device('cuda:3')
    
    yolo_v8n = YOLO('config/yolov8-KiTTi.yaml')
    
    if os.path.exists(yolov8_weights_path):
        print(f'YOLOv8 Model loaded')
        yolo_v8n.load(
            yolov8_weights_path
        )
    
    point_net = PointNetCls(num_classes=9)
    
    if os.path.exists(pointnet_weights_path):
        point_net.load_state_dict(
            torch.load(pointnet_weights_path)['model_state_dict']
        )
        
    point_net = point_net.feature_transform
    
    print(yolo_device, pointnet_device)
    
    yolo_v8n.to(yolo_device)
    point_net.to(pointnet_device)
    
    fuser_pipeline = Yolov8FuserPipeline(
        yolo_v8n, point_net, 
        yolo_device, pointnet_device,
        adaptive_fusion_kwargs, 
        image_resize
    )
        
    # # target_layers = [15, 18, 21]
    
    # # for idx in target_layers:
    # #     print(yolo_v8n.model.model[idx])
    # #     print('---------'*5)
    
    # print(yolo_v8n)
    # exit(1)
    
    dataset = Kitti2DObjectDetectDataset(
        lidar_dir="data/velodyne/training/velodyne",
        calibration_dir="data/calibration/training/calib",
        left_image_dir="data/left_images/training/image_2",
        labels_dir="data/labels/training/label_2"
    )
        
    dataloader = DataLoader(
        dataset, 
        batch_size=8,
        collate_fn=KittiLidarFusionCollateFn(
            image_resize=image_resize,
            apply_data_fusion=False,
            grid_sizes=grid_sizes,
            project_2d=True
        ),
        shuffle=True
    )
    
    # yolo.to('cuda')
    
    for data_items in dataloader:
        for k,v in data_items.items():
            if torch.is_tensor(v):                    
                # data_items[k] = v.to('cuda')     
                print(f'{k} {v.size()}')
                
        with torch.no_grad():
            outputs = fuser_pipeline(data_items['images'],
            data_items['raw_point_clouds'],
            data_items['proj2d_pc_mask'], 
            data_items['targets'])
            
            print(outputs['total_loss'])
            
        # exit(1)
        