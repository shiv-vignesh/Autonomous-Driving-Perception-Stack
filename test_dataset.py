import torch, time
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiLidarFusionCollateFn, KittiMLSFCollateFn, KitiiMLSFCollateAugment
from dataset_utils.enums import Enums
from model.mlsf_utils import box_center_to_corner
# from trainer.loss import compute_loss
# from model.yolo import Darknet

# from model.mlsf_yolo import MLSF
# from model.mlsf_mobilenet import MLSF
from model.mlsf_yolov8 import MLSFYolov8

if __name__ == "__main__":
    
    # yolo = Darknet(
    #     'config/yolov3-KiTTi.cfg'
    # )
    
    # image_resize = (416, 416)
    
    # mlsf = MLSF(config_path='config/yolov3-yolo_reduced_classes.cfg')
    mlsf = MLSFYolov8(
        yolov8_weights_path='yolov8n.pt',
        fusion_type='attention',
        weighted_fusion=True,
        num_fusion_blocks=2,
        num_classes=len(Enums.KiTTi_label2Id)
    )
    
    dataset = Kitti2DObjectDetectDataset(
        lidar_dir="data/KiTTi/training/velodyne",
        calibration_dir="data/KiTTi/training/calib",
        left_image_dir="data/KiTTi/training/image_2",
        labels_dir="data/KiTTi/training/label_2"

    )

    dataloader = DataLoader(
        dataset, 
        batch_size=2,
        collate_fn=KittiMLSFCollateFn(
            image_resize=(640, 640), 
            detection_head='yolo'
        ),
        shuffle=True
    )        
    
    for data_items in dataloader:
        
        # for k, v in data_items.items():
        #     if torch.is_tensor(v):
        #         print(f'{k} {v.shape}')
                
        # exit(1)
        
        mlsf(
            data_items['images'],
            data_items['lidar_depth_2d'],
            data_items['targets']
        )
        
        # image_path = data_items['image_paths'][0]        
        # for depth in mlsf.detection_depths:
        #     anchor_boxes = mlsf.anchor_info[depth]['default_boxes']
        #     anchor_corners = box_center_to_corner(anchor_boxes)
            
        #     print(anchor_corners)
            
        # exit(1)
        # total_loss, loss_components, detections = mlsf(
        #     data_items['images'],
        #     data_items['lidar_depth_2d'],
        #     data_items['targets'],
        # )
        
        # outputs = mlsf(
        #     data_items['images'],
        #     data_items['lidar_depth_2d'],
        #     data_items['targets'],
        # )        
        
        # print(outputs)

        # exit(1)        