import os
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm
import numpy as np
from terminaltables import AsciiTable

from model.mlsf_yolo import MLSFYolo
from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiMLSFCollateFn
from dataset_utils.enums import Enums
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
        collate_fn=KittiMLSFCollateFn(
            image_resize=test_dataset_kwargs['image_resize'],
            detection_head='yolo'
        ),
        shuffle=test_dataset_kwargs['shuffle']
    )
    
    return dataloader

def load_mlsf_yolo(model_kwargs:dict, weights_path:str):
    
    image_backbone_device = torch.device(model_kwargs['image_backbone_device']) if torch.cuda.is_available() else torch.device('cpu')
    lidar_backbone_device = torch.device(model_kwargs['lidar_backbone_device']) if torch.cuda.is_available() else torch.device('cpu')
    adaptive_fusion_device = torch.device(model_kwargs['adaptive_fusion_device']) if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed(model_kwargs['model_seed'])
        torch.cuda.manual_seed_all(model_kwargs['model_seed'])
        
    else:
        torch.manual_seed(model_kwargs['model_seed']) 
    
    mlsf = MLSFYolo(
        config_path=model_kwargs['cfg_file'], 
        image_backbone_device=image_backbone_device, 
        lidar_backbone_device=lidar_backbone_device,
        adaptive_fusion_device=adaptive_fusion_device, 
        apply_adaptive_fusion=model_kwargs['apply_adaptive_fusion'], 
        num_fusion_blocks=model_kwargs['num_fusion_blocks'], 
        fusion_type=model_kwargs['fusion_type'], 
        weighted_fusion=model_kwargs['weighted_fusion']
    )    
    
    if os.path.exists(weights_path):
        print(f'Weights Loaded: {weights_path}')
        mlsf.to(
            torch.device('cpu')
        )        

        mlsf.load_state_dict(
            torch.load(weights_path)
        )
        
        mlsf.image_backbone.to(mlsf.image_backbone_device)
        mlsf.lidar_backbone.to(mlsf.lidar_backbone_device)
        mlsf.adaptive_fusion_module.to(mlsf.adaptive_fusion_device)        
        
        return mlsf
    
    else:
        exit(1)
        
def test(mlsf:MLSFYolo, dataloader:torch.utils.data.DataLoader, image_resize, output_dir):
    
    mlsf.eval()
    test_iter = tqdm(dataloader)
    
    image_paths = []
    image_detections = []    
    
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    img_size = mlsf.image_backbone.hyperparams['height']    
    
    for batch_idx, data_items in enumerate(test_iter):
        with torch.no_grad():
            loss, _, outputs = mlsf(
                data_items['images'],
                data_items['lidar_depth_2d'] if mlsf.use_lidar_backbone else None,
                data_items['targets']
            )

        targets = data_items['targets'].cpu()
        labels += targets[:, 1] #[class_id]   
        
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        
        anchor_grids = [yolo_layer.anchor_grid for yolo_layer in mlsf.image_backbone.yolo_layers]
        outputs = apply_sigmoid_activation(outputs, data_items['images'].size(2), anchor_grids)                
        outputs = non_max_suppression(outputs)
        
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=0.5)
        
        image_detections.extend(outputs)
        image_paths.extend(data_items['image_paths'])
        
        if (batch_idx + 1) % 10 == 0:
        
            if image_detections:
                class_names = list(Enums.KiTTi_label2Id.keys())  
                draw_and_save_output_images(
                    image_detections, image_paths, 
                    image_resize[0],
                    f'{output_dir}', class_names
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
        "mlsf_yolo_kwargs":{
            "cfg_file":"config/yolov3-yolo_reduced_classes.cfg",
            "image_channels":3, 
            "lidar_channels":3, 
            "image_backbone_device":"cuda:16",
            "lidar_backbone_device":"cuda:16",
            "adaptive_fusion_device":"cuda:13",
            "apply_adaptive_fusion":True, 
            "fusion_type":"attention",
            "num_fusion_blocks":2, 
            "weighted_fusion":False,
            "model_seed":101
        },
        "kitti_validation_dataset_kwargs":{
            "lidar_dir":"data/KiTTi/validation/velodyne",
            "calibration_dir":"data/KiTTi/validation/calib",
            "left_image_dir":"data/KiTTi/validation/image_2",
            "right_image_dir":None,
            "labels_dir":"data/KiTTi/validation/label_2",
            "shuffle":False,
            "apply_augmentation":False, 
            "batch_size":12, 
            "image_resize":[416, 416]
        }, 
        "output_dir":"MLSF-YOLO-Attention-WeightedGated-FocalLoss/best-model/detections_2"
    }

    model_path = "MLSF-YOLO-Attention-WeightedGated-FocalLoss/best-model/best-model.pt"
    
    mlsf = load_mlsf_yolo(
        test_kwargs['mlsf_yolo_kwargs'], model_path
    )
    
    dataloader = create_dataloader(
        test_kwargs['kitti_validation_dataset_kwargs']
    )
    
    if not os.path.exists(test_kwargs['output_dir']):
        os.makedirs(test_kwargs['output_dir'])
        
    test(
        mlsf=mlsf,
        dataloader=dataloader, 
        image_resize=test_kwargs["kitti_validation_dataset_kwargs"]['image_resize'],
        output_dir=test_kwargs['output_dir']
    )
    