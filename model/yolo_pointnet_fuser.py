from typing import Iterable
from collections import defaultdict
import math
import numpy as np
import torch 

# from .point_net import PointNetBackbone
from .t_net import TransformNet
from .yolo import Darknet

class ImprovedAdaptiveFusion(torch.nn.Module):
    def __init__(self, yolo_grid_channels: int, lidar_grid_channels: int,
                 device: torch.device,
                 fusion_type: str = 'residual',
                 transform_image_features:bool=True, 
                 alpha: float = 0.5):
        super(ImprovedAdaptiveFusion, self).__init__()
        
        # Dimension reduction for both modalities
        self.yolo_dim_reduce = torch.nn.Conv2d(
            yolo_grid_channels, yolo_grid_channels // 2, kernel_size=1
        ).to(device)
        
        self.lidar_dim_reduce = torch.nn.Conv2d(
            lidar_grid_channels, yolo_grid_channels // 2, kernel_size=1
        ).to(device)
        
        # Cross-attention layers
        self.query_conv = torch.nn.Conv2d(yolo_grid_channels // 2, yolo_grid_channels // 2, kernel_size=1).to(device)
        self.key_conv = torch.nn.Conv2d(yolo_grid_channels // 2, yolo_grid_channels // 2, kernel_size=1).to(device)
        self.value_conv = torch.nn.Conv2d(yolo_grid_channels // 2, yolo_grid_channels // 2, kernel_size=1).to(device)
        
        # Output projection
        self.output_proj = torch.nn.Conv2d(
            yolo_grid_channels // 2, yolo_grid_channels, kernel_size=1
        ).to(device)
        
        # Layer normalization for better training stability
        self.norm1 = torch.nn.LayerNorm([yolo_grid_channels // 2]).to(device)
        self.norm2 = torch.nn.LayerNorm([yolo_grid_channels]).to(device)
        
        self.dropout = torch.nn.Dropout(0.1).to(device)
        self.fusion_type = fusion_type
        self.alpha = alpha
        self.yolo_grid_channels = yolo_grid_channels
        
        self.apply(self.weights_init_normal)
    
    @staticmethod
    def weights_init_normal(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
                          
    def forward(self, yolo_grid_features: torch.Tensor, lidar_grid_features: torch.Tensor):
        
        # print(yolo_grid_features.shape)
        
        bs, c, h, w = yolo_grid_features.shape
        
        # Dimension reduction
        yolo_feat = self.yolo_dim_reduce(yolo_grid_features)
        lidar_feat = self.lidar_dim_reduce(lidar_grid_features)
                
        # Apply layer norm
        yolo_feat = self.norm1(yolo_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        lidar_feat = self.norm1(lidar_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Compute Q, K, V
        queries = self.query_conv(yolo_feat)
        keys = self.key_conv(lidar_feat)
        values = self.value_conv(lidar_feat)
        
        # Reshape for attention
        queries = queries.view(bs, -1, h * w).permute(0, 2, 1)  # (bs, h*w, c)
        keys = keys.view(bs, -1, h * w)  # (bs, c, h*w)
        values = values.view(bs, -1, h * w).permute(0, 2, 1)  # (bs, h*w, c)
        
        # Compute attention scores
        attn_weights = torch.matmul(queries, keys) / math.sqrt(self.yolo_grid_channels // 2)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        fusion_features = torch.matmul(attn_weights, values)
        fusion_features = fusion_features.permute(0, 2, 1).view(bs, -1, h, w)
        
        # Project back to original dimensions
        fusion_features = self.output_proj(fusion_features)
        fusion_features = self.norm2(fusion_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        if self.fusion_type == "residual":
            return (yolo_grid_features + self.alpha * fusion_features), yolo_feat, lidar_feat
            # return fusion_features, yolo_feat, lidar_feat
        else:  # multiplicative
            return (yolo_grid_features * torch.sigmoid(fusion_features)), yolo_feat, lidar_feat

class AdaptiveFusion(torch.nn.Module):
    def __init__(self, yolo_grid_channels:int, lidar_grid_channels:int,                  
                 device:torch.device, 
                 transform_image_features:bool=True, 
                 fusion_type:str='residual', #[residual or multiplicative]
                 alpha:float=0.5   
                 ):
        super(AdaptiveFusion, self).__init__()
        
        self.lidar_transform = torch.nn.Conv2d(
            lidar_grid_channels, yolo_grid_channels, kernel_size=1, stride=1, bias=False 
        ).to(device)
        
        self.image_transform = torch.nn.Conv2d(
            yolo_grid_channels, yolo_grid_channels, kernel_size=1, stride=1, bias=False
        ).to(device)
        
        self.yolo_grid_channels = yolo_grid_channels
        self.yolo_device = device
        
        self.transform_image_features = transform_image_features
        self.fusion_type = fusion_type
        self.alpha = alpha
        
        self.apply(self.weights_init_normal)

    @staticmethod
    def weights_init_normal(m):
        """
        Applies normal weight initialization to convolutional and batch normalization layers.
        """
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, yolo_grid_features:torch.tensor, lidar_grid_features:torch.tensor):        
        
        yolo_grid_features = self.image_transform(yolo_grid_features) if self.transform_image_features else yolo_grid_features
        lidar_grid_features = self.lidar_transform(lidar_grid_features)
        
        attn_grid_map = (yolo_grid_features * lidar_grid_features) / math.sqrt(self.yolo_grid_channels)
        bs, w, h, hidden = attn_grid_map.shape
        
        attn_grid_map = torch.sigmoid(input=attn_grid_map.view(bs, -1)).reshape(bs, w, h, hidden)
        
        if self.fusion_type == "residual":
            return yolo_grid_features + self.alpha * attn_grid_map
        
        elif self.fusion_type == 'multiplicative':
            return yolo_grid_features * attn_grid_map 

class FuserPipeline(torch.nn.Module):
    def __init__(self, yolo:Darknet, pointnet:TransformNet, 
                 yolo_device:torch.device, point_net_device:torch.device, 
                 adaptive_fusion_kwargs:dict):
        super(FuserPipeline, self).__init__()
        
        self.yolo = yolo
        self.pointnet = pointnet
        
        self.yolo_device = yolo_device
        self.point_net_device = point_net_device
        self.adaptive_fusion_device = yolo_device        
        
        self.fusion_gates = torch.nn.ModuleDict({
            'fusion_13x13':ImprovedAdaptiveFusion(1024, 1024, self.adaptive_fusion_device, **adaptive_fusion_kwargs),
            'fusion_26x26':ImprovedAdaptiveFusion(512, 1024, self.adaptive_fusion_device, **adaptive_fusion_kwargs),
            'fusion_52x52':ImprovedAdaptiveFusion(256, 1024, self.adaptive_fusion_device, **adaptive_fusion_kwargs)
            
        })
        
        self.image_size = self.yolo.hyperparams['height'], self.yolo.hyperparams['width']
    
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
            
    def forward(self, images:torch.tensor, raw_point_clouds:torch.tensor, proj2d_pc_mask:Iterable[dict]):                
        
        yolo_backbone_features = self.yolo.forward_backbone(
            images.to(self.yolo_device)
        ) #List[(bs, 1024, 13, 13), (bs, 512, 26, 26), (bs, 256, 52, 52)]     
        
        #input : raw_point_clouds is most likely voxelized. Check __collatefn()__
        # output: torch.Size([bs, 1024, num_points])
        point_net_features = self.pointnet(
            raw_point_clouds.to(self.point_net_device), #(bs, 3, num_points)
            return_global=True
        )
        

        lidar_grid_features = self.project_to_grid(point_net_features, proj2d_pc_mask)
        
        del point_net_features
        del raw_point_clouds
        
        fused_features_list = []       
        yolo_features_list, lidar_features_list = [], []
        
        
        for idx, (_, fusion_module) in enumerate(self.fusion_gates.items()):        
            fused_features, yolo_feat, lidar_feat = fusion_module(yolo_backbone_features[idx].to(self.adaptive_fusion_device), 
                                           lidar_grid_features[idx].to(self.adaptive_fusion_device))
            
            yolo_features_list.append(yolo_feat)
            lidar_features_list.append(lidar_feat)
            
            fused_features_list.append(fused_features.to(self.yolo_device))

        del lidar_grid_features
        
        return self.yolo.forward_detection_head(fused_features_list, images.shape[2]), yolo_features_list, lidar_features_list
                  
    def save_model_ckpts(self, output_dir:str, cur_epoch:int):
        
        torch.save(
            self.yolo.state_dict(), f'{output_dir}/yolo_weights_{cur_epoch}.pth'
        )
        
        torch.save(
            self.pointnet.state_dict(), f'{output_dir}/pointnet_{cur_epoch}.pth'
        )
        
        torch.save(
            self.fusion_gates.state_dict(), f'{output_dir}/fusion_gates{cur_epoch}.pth'
        )    
        
    def load_model_ckpts(self, output_dir:str, cur_epoch:int):
        
        self.yolo.load_state_dict(
            torch.load(f'{output_dir}/yolo_weights_{cur_epoch}.pth', map_location=self.yolo_device)
        )
        
        self.pointnet.load_state_dict(
            torch.load(f'{output_dir}/pointnet_{cur_epoch}.pth', map_location=self.point_net_device)
        )
        
        self.fusion_gates.load_state_dict(
            torch.load(f'{output_dir}/fusion_gates{cur_epoch}.pth', map_location=self.adaptive_fusion_device)
        )
        