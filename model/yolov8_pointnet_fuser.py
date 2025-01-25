from typing import Iterable
from collections import defaultdict
import math
import numpy as np
import torch 

from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

# from .point_net import PointNetBackbone
from .t_net import TransformNet
from .yolov3_pointnet_fuser import ImprovedAdaptiveFusion


class Yolov8FuserPipeline(torch.nn.Module):
    def __init__(self, yolov8:YOLO, pointnet:TransformNet, 
                 yolo_device:torch.device, point_net_device:torch.device, 
                 adaptive_fusion_kwargs:dict, image_size:tuple):
        super(Yolov8FuserPipeline, self).__init__()
        
        self.yolo = yolov8.model
        self.pointnet = pointnet

        self.yolo_device = yolo_device
        self.point_net_device = point_net_device
        self.adaptive_fusion_device = yolo_device   
        
        self.fusion_gates = torch.nn.ModuleDict({
            'fusion_20x20':ImprovedAdaptiveFusion(256, 1024, self.adaptive_fusion_device, **adaptive_fusion_kwargs),
            'fusion_40x40':ImprovedAdaptiveFusion(128, 1024, self.adaptive_fusion_device, **adaptive_fusion_kwargs),
            'fusion_80x80':ImprovedAdaptiveFusion(64, 1024, self.adaptive_fusion_device, **adaptive_fusion_kwargs)            
        })
        
        self.feature_maps = {} #layer_idx:tensor
        self.detection_map = None
        
        self.image_size = image_size
        self.feature_layers = [15, 18, 21]
        self.detection_layer = self.yolo.model[-1]
        
        for idx in self.feature_layers:
            layer = self.yolo.model[idx]
            layer.register_forward_hook(self._hook_feature_map())
            
        self.detection_layer.register_forward_hook(
            self._hook_detection()
        )
        
        self.detection_loss = v8DetectionLoss(self.yolo)     
        self.num_classes = self.yolo.yaml['nc']    
                
        # self.detection_loss.hyp['box'] = 7.5  # default value for box loss gain
        # self.detection_loss.hyp['cls'] = 0.5  # default value for cls loss gain
        # self.detection_loss.hyp['dfl'] = 1.5  # default value for dfl loss gain
        
    def _hook_feature_map(self):
        def hook(module, input, output):
            _, _, h, w = output.shape
            self.feature_maps[f'{h}x{w}'] = output
            
        return hook
    
    def _hook_detection(self):
        def hook(module, input, output):
            self.detection_map = output
        return hook    

    def project_to_grid(self, point_net_features:torch.tensor, 
                        proj2d_pc_mask:Iterable[dict]):
        
        _, latent_dim, num_points = point_net_features.shape
        
        batch_lidar_grid_features = defaultdict(list)
        
        for batch_idx, point_net_feature in enumerate(point_net_features):
            for grid_size in proj2d_pc_mask[batch_idx]:
                
                lidar_grid = torch.zeros(latent_dim, grid_size[0], grid_size[1], device=point_net_features.device)
                count_grid = torch.zeros(grid_size[0], grid_size[1], device=point_net_features.device)
                ''' 
                valid_mask_x - [raw_points] bool tensor
                valid_mask_y - [raw_points] bool tensor
                valid_grid_coords - [num_valid_points, 2] (comprises of x & y values) 
                obtained from transforming raw_point cloud --> img --> grid space
                '''
                
                if "valid_indices" in proj2d_pc_mask[batch_idx][grid_size]:
                    valid_indices = proj2d_pc_mask[batch_idx][grid_size]['valid_indices']
                    
                if "count_grid" in proj2d_pc_mask[batch_idx][grid_size]:
                    count_grid = proj2d_pc_mask[batch_idx][grid_size]['count_grid']
                
                else:                
                    valid_mask_x = proj2d_pc_mask[batch_idx][grid_size]['valid_mask_x']
                    valid_mask_y = proj2d_pc_mask[batch_idx][grid_size]['valid_mask_y']
                    valid_indices = valid_mask_x & valid_mask_y #(num_valid_points)
                
                valid_grid_coords = proj2d_pc_mask[batch_idx][grid_size]['valid_grid_coords']
                                                
                if valid_indices.sum() > 0:
                    
                    valid_pointnet_features = point_net_feature[:, valid_indices]       
                    
                    lidar_grid[:, 
                               valid_grid_coords[:, 0],
                               valid_grid_coords[:, 1]] += valid_pointnet_features
                    
                    lidar_grid /= (count_grid.unsqueeze(0).to(point_net_features.device) + 1e-6)                    
                    lidar_grid = lidar_grid.clamp(min=0)
                    lidar_grid[lidar_grid == 0] = 1e-5
                    lidar_grid = torch.log1p(lidar_grid)
                
                    batch_lidar_grid_features[grid_size].append(lidar_grid)                
            
        return [torch.stack(batch_grid_feature) for batch_grid_feature in batch_lidar_grid_features.values()]
        
    def forward(self, images:torch.tensor, 
                raw_point_clouds:torch.tensor, proj2d_pc_mask:Iterable[dict], 
                targets=None):
        
        point_net_features = self.pointnet(
            raw_point_clouds.to(self.point_net_device), #(bs, 3, num_points)
            return_global=True
        )
        
        yolo_outputs = self.yolo(images.to(self.yolo_device))        
        lidar_grid_features = self.project_to_grid(point_net_features, proj2d_pc_mask)
                        
        for idx, (_, fusion_module) in enumerate(self.fusion_gates.items()):
            lidar_grid = lidar_grid_features[idx]
            
            _, _, h, w = lidar_grid.shape            
            yolo_grid = self.feature_maps[f'{h}x{w}']
                        
            fused_features, _, _ = fusion_module(yolo_grid.to(self.adaptive_fusion_device), 
                                           lidar_grid.to(self.adaptive_fusion_device))
                        
            self.feature_maps[f'{h}x{w}'] = fused_features.to(self.yolo_device)  
        
        detections = self.detection_layer(list(self.feature_maps.values()))   
                        
        if targets is not None:
            batch_idx = targets[:, 0].unsqueeze(1).long()  # Shape: (N, 1)
            cls = targets[:, 1].unsqueeze(1)  # Shape: (N, 1)
            bboxes = targets[:, 2:]  # Shape: (N, 4)
            
            restructured_targets = {
                'batch_idx': batch_idx,
                'cls': cls,
                'bboxes': bboxes
            }
            
            total_loss, loss_components = self.compute_yolov8_loss(detections, restructured_targets)
            
        else:
            total_loss = None 
            loss_components = None
                
        return {
            # 'predictions':detections[0], #Iterable[tensor(8400, 84) * batch_size]
            'feature_maps':detections, #Iterable[[bs, 144, 80, 80], [bs, 144, 40, 40], [bs, 144, 20, 20]]
            'total_loss':total_loss, 
            'loss_components':loss_components
        }
    
    def compute_yolov8_loss(self, predictions:Iterable[torch.tensor], batch:dict):
        
        # patching code; 
        # from ultralytics.utils.loss import v8DetectionLoss
                
        loss = torch.zeros(3, device=self.detection_loss.device)  # box, cls, dfl
        feats = predictions[1] if isinstance(predictions, tuple) else predictions
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.detection_loss.no, -1) for xi in feats], 2).split(
            (self.detection_loss.reg_max * 4, self.detection_loss.nc), 1
        )        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.detection_loss.device, dtype=dtype) * self.detection_loss.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.detection_loss.stride, 0.5)        
        
        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.detection_loss.preprocess(targets.to(self.detection_loss.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)        
        
        # Pboxes
        pred_bboxes = self.detection_loss.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.detection_loss.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.detection_loss.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.detection_loss.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )  
            
        box_gain = 7.5
        cls_gain = 0.5
        dfl_gain = 1.5
        
        loss[0] *= box_gain  # box gain
        loss[1] *= cls_gain  # cls gain
        loss[2] *= dfl_gain  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)        