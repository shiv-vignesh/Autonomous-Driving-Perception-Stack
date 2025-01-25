import os
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm
import numpy as np
from terminaltables import AsciiTable

from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiLidarFusionCollateFn
from dataset_utils.enums import Enums
from model.yolo import Darknet
from model.point_net import PointNetBackbone
from model.t_net import PointNetCls 
from model.yolov3_pointnet_fuser import FuserPipeline
from model.yolo_utils import xywh2xyxy, reshape_outputs, apply_sigmoid_activation, non_max_suppression, rescale_boxes, get_batch_statistics, ap_per_class
from test_yolo import draw_and_save_output_images

def print_eval_stats(metrics_output, class_names, output_dir:str,verbose=True):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP", "precision", "recall", "F1"]]
            for i, c in enumerate(ap_class):
                # ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                ap_table.append([
                    c, 
                    class_names[c],
                    f'{AP[i]:.5f}',
                    f'{precision[i]:.5f}',
                    f'{recall[i]:.5f}',
                    f'{f1[i]:.5f}'
                ])
            
            table_string = AsciiTable(ap_table).table
            
            print(f'---------- mAP per Class----------')
            print(f'{table_string}')
        
            print(f'---------- Total mAP {AP.mean():.5f} ----------')
            
            with open(f'{output_dir}/metrics.txt', 'w+') as f:
                f.write(table_string)
            
    else:
        print("---- mAP not measured (no detections found by model) ----") 

def create_dataloader(test_dataset_kwargs:dict):
    dataset = Kitti2DObjectDetectDataset(
        lidar_dir=test_dataset_kwargs['lidar_dir'],
        calibration_dir=test_dataset_kwargs['calibration_dir'],
        left_image_dir=test_dataset_kwargs['left_image_dir'],
        right_image_dir=test_dataset_kwargs['right_image_dir'],
        labels_dir=test_dataset_kwargs['labels_dir']
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=test_dataset_kwargs['batch_size'],
        collate_fn=KittiLidarFusionCollateFn(
            image_resize=test_dataset_kwargs['image_resize'],
            precomputed_voxel_dir="",
            precomputed_proj2d_dir="",
            apply_augmentation=False,
            project_2d=True, 
            voxelization=True, 
            categorize_labels=True       
        ),
        shuffle=test_dataset_kwargs['shuffle']
    )
    
    return dataloader

def load_model(config_path:str, adaptive_fusion_kwargs, yolo_device_id:str, pointnet_device_id)->FuserPipeline:
    yolo_device = torch.device(yolo_device_id) if torch.cuda.is_available() else torch.device("cpu")        
    point_net_device = torch.device(pointnet_device_id) if torch.cuda.is_available() else torch.device("cpu")            
    
    yolo = Darknet(config_path).to(yolo_device)
    point_net = PointNetCls(num_classes=9).feature_transform.to(point_net_device)
    
    fuser_pipeline = FuserPipeline(yolo, point_net, 
                                   yolo_device, point_net_device, adaptive_fusion_kwargs)   
    
    return fuser_pipeline 

def test(config_path:str, spatial_fusion_ckpt_dir:str, spatial_fusion_ckpt:int,
        yolo_device_id:str, pointnet_device_id:str, 
        test_dataset_kwargs:dict, 
        adaptive_fusion_kwargs:dict, 
        output_dir:str):    
    
    fuser_pipeline = load_model(config_path, adaptive_fusion_kwargs, yolo_device_id, pointnet_device_id)    
    test_dataloader = create_dataloader(test_dataset_kwargs)
    
    if os.path.exists(spatial_fusion_ckpt_dir):
        fuser_pipeline.load_model_ckpts(spatial_fusion_ckpt_dir, spatial_fusion_ckpt)   
        
    else:
        print(f'File not Founds')
        exit(1)
        
    if not os.path.exists(f'{output_dir}'):
        os.makedirs(output_dir)

    if not os.path.exists(f'{output_dir}/detections'):
        os.makedirs(f'{output_dir}/detections')   
        
    fuser_pipeline.eval()
    
    image_paths = []
    image_detections = []
    
    test_iter = tqdm(test_dataloader)
    
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    img_size = fuser_pipeline.yolo.hyperparams['height']      
    
    for batch_idx, data_items in enumerate(test_iter):
       
        targets = data_items['targets'].cpu()
        labels += targets[:, 1] #[class_id] 
                    
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size        
        
        with torch.no_grad():
            outputs, _, _ = fuser_pipeline(
                data_items['images'],
                data_items['raw_point_clouds'],
                data_items['proj2d_pc_mask'])      
            
        anchor_grids = [yolo_layer.anchor_grid for yolo_layer in fuser_pipeline.yolo.yolo_layers]            
        outputs = reshape_outputs(outputs)            
        outputs = apply_sigmoid_activation(outputs, data_items['images'].size(2), anchor_grids)                
        outputs = non_max_suppression(outputs)           
        
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=0.4)  
        image_detections.extend(outputs)
        image_paths.extend(data_items['image_paths'])
        
        if (batch_idx + 1) % 2 == 0:
                
            class_names = list(Enums.KiTTi_label2Id.keys())         
            draw_and_save_output_images(
                image_detections, image_paths, 
                test_dataset_kwargs['image_resize'][0],
                f'{output_dir}/detections', class_names
            )
            
            image_detections = []
            image_paths = []

    if image_detections:
        class_names = list(Enums.KiTTi_label2Id.keys())  
        draw_and_save_output_images(
            image_detections, image_paths, 
            test_dataset_kwargs['image_resize'][0],
            f'{output_dir}/detections', class_names
        )
        
        image_detections = []
        image_paths = []       
            
    print(f'Detection Finished! Computing Metrics')
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]            

    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels) 
    
    class_names = list(Enums.KiTTi_label2Id.keys())    
    print_eval_stats(metrics_output, class_names, output_dir)        

if __name__ == "__main__":
    
    test_kwargs = {
        "config_path":"config/yolov3-yolo_reduced_classes.cfg",
        "spatial_fusion_ckpt_dir" : 'Robust-Spatial-Fusion-Pipeline-2/best-model',
        "spatial_fusion_ckpt":12,
        "yolo_device_id":"cuda:2",
        "pointnet_device_id":"cuda:4",
        "adaptive_fusion_kwargs":{
            "transform_image_features":False, 
            "fusion_type":"residual",
            "alpha":1.0
        },        
        "test_dataset_kwargs":{
            "lidar_dir":"data/velodyne/validation/velodyne",
            "calibration_dir":"data/calibration/validation/calib",
            "left_image_dir":"data/left_images/validation/image_2",
            "right_image_dir":None,
            "labels_dir":"data/labels/validation/label_2",
            "batch_size":8,
            "shuffle":False, 
            "image_resize":[416, 416]
        },
        "output_dir":"Robust-Spatial-Fusion-Pipeline-2/detections-3"
    }
    
    test(**test_kwargs)
    
        