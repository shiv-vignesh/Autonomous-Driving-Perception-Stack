import os
import cv2 
import albumentations
import numpy as np

from typing import List, Dict

import torch
from torch.utils.data import Dataset

from .enums import Enums

class Kitti2DObjectDetectDataset(Dataset):
    
    def __init__(self, lidar_dir:str, 
                calibration_dir:str, 
                left_image_dir:str=None, 
                right_image_dir:str=None,
                labels_dir:str=None, 
                dataset_type:str="train"):
        
        self.left_image_dir = left_image_dir 
        self.right_image_dir = right_image_dir
                        
        if not bool(self.left_image_dir) and not bool(self.right_image_dir):
            raise Exception(f'Both Left and Right Images cannot be {self.left_image_dir}')
            
            
        self.calibration_dir = calibration_dir
        self.lidar_dir = lidar_dir
        self.labels_dir = labels_dir 
        self.dataset_type = dataset_type
        
        self.lidar_files = os.listdir(self.lidar_dir)
        self.left_image_files = os.listdir(self.left_image_dir)
        self.right_image_files = os.listdir(self.right_image_dir)
        self.calibration_files = os.listdir(self.calibration_dir)
        self.label_files = os.listdir(self.labels_dir)
        
    def __len__(self):
        return len(
            self.lidar_files
        )
        
    def __getitem__(self, idx):
        
        lidar_file = self.lidar_files[idx]
        _id = lidar_file.split('.')[0]
        
        return {
            'lidar_file_path':f'{self.lidar_dir}/{lidar_file}',
            'calibration_file_path':f'{self.calibration_dir}/{_id}.txt',
            'left_image_file_path': f'{self.left_image_dir}/{_id}.png' if self.left_image_dir is not None else None,
            'right_image_file_path':f'{self.right_image_dir}/{_id}.png' if self.right_image_dir is not None else None,
            'label_file_path':f'{self.labels_dir}/{_id}.txt' if self.labels_dir is not None else None
        }
        
class KittiLidarFusionCollateFn(object):
    
    def __init__(self, image_resize:list, transformation=None, clip_distance:float=2.0):
        '''
        TODO, 
        1. Transformation Function from https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/augmentations.py#L29
        2. Combine Left and Right Images 
        3. 
        '''
        self.image_resize = image_resize
        self.transformation = transformation
        self.clip_distance = clip_distance
        
        if self.transformation is None:
            self.transformation = albumentations.Compose(
                [albumentations.Resize(height=self.image_resize[0], width=self.image_resize[1], always_apply=True)],
                bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )
            
            self.image_only_transformation = albumentations.Compose(
                [albumentations.Resize(height=self.image_resize[0], width=self.image_resize[1], always_apply=True)]
            )

    def read_velodyne_bin(self, file_path):
        # Load the .bin file
        point_cloud = np.fromfile(file_path, dtype=np.float32)
            
        # Reshape the array to (N, 4), where N is the number of points
        point_cloud = point_cloud.reshape(-1, 4)    
        # Extract x, y, z coordinates
        return point_cloud

    def read_calibration_file(self, calib_file_path):
        calibration_dict = {}
        
        with open(calib_file_path, 'r') as f:
            for line in f.readlines():
                if line != '\n':
                    key, value = line.split(':')
                    calibration_dict[key.strip()] = np.fromstring(
                        value, sep=' '
                    )
                    
        return calibration_dict

    def read_image(self, image_path:str):
        return cv2.imread(image_path)
    
    def read_label_file(self, label_file_path:str):
        '''
        #Values    Index    Name      Description
        ----------------------------------------------------------------------------
        1        0       type      Describes the type of object: 'Car', 'Van', 'Truck',
                                    'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                    'Misc' or 'DontCare'
        1        1       truncated Float from 0 (non-truncated) to 1 (truncated), where
                                    truncated refers to the object leaving image boundaries
        1        2       occluded  Integer (0,1,2,3) indicating occlusion state:
                                    0 = fully visible, 1 = partly occluded
                                    2 = largely occluded, 3 = unknown
        1        3       alpha     Observation angle of object, ranging [-pi..pi]
        4        4-7       bbox      2D bounding box of object in the image (0-based index):
                                    contains left, top, right, bottom pixel coordinates
        3        8-10       dimensions 3D object dimensions: height, width, length (in meters)
        3        11-13       location  3D object location x,y,z in camera coordinates (in meters)
        1        14       rotation_y Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1        15       score     Only for results: Float, indicating confidence in
                                    detection, needed for p/r curves, higher is better.

        '''
        class_labels = []
        bboxes = []

        with open(label_file_path, 'r') as file:
            for line in file:                
                parts = line.strip().split() # Split the line into parts

                # Extract class label and bounding box coordinates
                obj_type = parts[0]  # First element is the object type
                if obj_type not in Enums.KiTTi_label2Id:
                    continue  # Skip invalid object types

                # Bounding box coordinates
                left = float(parts[4])  # left
                top = float(parts[5])  # top
                right = float(parts[6])  # right
                bottom = float(parts[7])  # bottom

                class_id = Enums.KiTTi_label2Id[obj_type]
                
                class_labels.append(class_id)
                bboxes.append([left, top, right, bottom])  
                
        return class_labels, bboxes             
    
    def transform_lidar_points(self, point_cloud_array:np.array, calibration_dict:dict):

        # ignoring the last_dim as it corresponds to intensity.
        if point_cloud_array.shape[-1] > 3:
            point_cloud_array = point_cloud_array[:, :3]    #(N, 3)
        
        #reshaping rectification matrix from (12,) to (3,3)
        r0_rect = calibration_dict['R0_rect'].reshape(3,3) #(3,3)
        r0_rect_homo = np.vstack([r0_rect, [0, 0, 0]]) #(4,3)
        r0_rect_homo = np.column_stack([r0_rect_homo, [0, 0, 0, 1]]) #(4,4)
        
        # reshaping projection_matrix from (12,) to (3,4)
        proj_mat = calibration_dict['P2'].reshape(3,4) 
        
        # reshaping Tr_velo_to_cam from (12,) to (3,4)
        v2c = calibration_dict['Tr_velo_to_cam'].reshape(3,4)
        v2c = np.vstack(
            (v2c, [0, 0, 0, 1])
        ) #(4,4)    
        
        p_r0 = np.dot(proj_mat, r0_rect_homo) # (3, 4)
        p_r0_rt = np.dot(p_r0, v2c) #(3, 4)
        
        point_cloud_array = np.column_stack(
            [point_cloud_array, np.ones((point_cloud_array.shape[0], 1))]
        ) # (N, 4)
        
        #(3, 4) dot (4, N) ---> (3, N) ---> (N, 3)
        p_r0_rt_x = np.dot(
            p_r0_rt, point_cloud_array.T
        ).T 
        
        # The transformed coordinates are for LIDAR (u, v, z) to (u', v', z') in Image. Normalize by depth (z')
        p_r0_rt_x[:, 0] /= p_r0_rt_x[:, -1]
        p_r0_rt_x[:, 1] /= p_r0_rt_x[:, -1]
        
        return p_r0_rt_x[:, :2], p_r0_rt_x[:, -1]        
        
    def create_maps(self, projected_points:np.array, y_max:int, x_max:int, intensities:np.array, depths:np.array):

        reflectance_map = np.zeros((y_max, x_max), dtype=np.float32)
        depth_map = np.zeros((y_max, x_max), dtype=np.float32)   
        
        for i in range(len(projected_points)):
            x, y = int(projected_points[i, 0]), int(projected_points[i, 1])
            if 0 <= x < x_max and 0 <= y < y_max:
                # Use maximum intensity for reflectance
                reflectance_map[y, x] = max(reflectance_map[y, x], intensities[i])
                # Use the closest depth value
                if depth_map[y, x] == 0:  # if depth is not set yet
                    depth_map[y, x] = depths[i]

        # Normalize reflectance map for visualization
        reflectance_map = (reflectance_map / np.max(reflectance_map) * 255).astype(np.uint8)
        depth_map = (depth_map / np.max(depth_map) * 255).astype(np.uint8)
        
        depth_map_filtered = cv2.bilateralFilter(depth_map, d=9, sigmaColor=75, sigmaSpace=75)
        reflectance_map_filtered = cv2.bilateralFilter(reflectance_map, d=9, sigmaColor=75, sigmaSpace=75)
        
        return depth_map_filtered, reflectance_map_filtered
    
    def preprocess(self, lidar_point_cloud:np.array, calibration_dict:dict, image_array:np.array):
        
        points_2d, depths = self.transform_lidar_points(
            lidar_point_cloud, calibration_dict
        )        
        
        x_min, y_min = 0, 0
        x_max, y_max = image_array.shape[1], image_array.shape[0]
        
        fov_inds = (
                (points_2d[:, 0] < x_max)
                & (points_2d[:, 0] >= x_min)
                & (points_2d[:, 1] < y_max)
                & (points_2d[:, 1] >= y_min)
        )    
        
        fov_inds = fov_inds & (
                    lidar_point_cloud[:, 0] > self.clip_distance)      
        
        projected_points = points_2d[fov_inds]
        
        depth_map, reflectance_map = self.create_maps(
            projected_points, y_max, x_max, 
            lidar_point_cloud[fov_inds, -1],
            lidar_point_cloud[fov_inds, -2]            
        )         
        
        return depth_map, reflectance_map
    
    def transform_sample(self, image:np.array, label_bboxes:np.array=None, class_labels:np.array=None):                        
        
        if label_bboxes is not None and class_labels is not None:        
            transformed_dict = self.transformation(
                image=image, bboxes=label_bboxes, class_labels=class_labels
            )
        
        else:
            transformed_dict = self.image_only_transformation(
                image=image
            )            
        
        return transformed_dict
    
    def prepare_targets(self, batch_idx:int, class_labels:list, class_bboxes:list):

        targets = []
        for bbox, class_id in zip(class_bboxes, class_labels):        
            left, top, right, bottom = bbox

            x_center = (left + right) / 2 / self.image_resize[1]
            y_center = (top + bottom) / 2 / self.image_resize[0]
            width = (right - left) / self.image_resize[1]
            height = (bottom - top) / self.image_resize[0]

            targets.append([
                batch_idx, class_id, x_center, y_center, width, height
            ])
            
        return targets
            
    def __call__(self, batch_data_filepaths:List[Dict]):
        
        batch_data_items = {
            "images": [], #list of tensors --> stack --> tensor (bs, n_c, h, w),
            "targets": [],
            "image_paths":[]
            # "bboxes":[], # list of list of list [bs * [num_labels * [x,y,x,y] ]] (inner list of 4 elements)
            # "class_labels":[] # list of list of class_labels [bs * [num_labels] ] (inner list is class_ids)
        }
        
        for idx, file_path_dict in enumerate(batch_data_filepaths):
            
            lidar_file_path = file_path_dict['lidar_file_path']
            calibration_file_path = file_path_dict['calibration_file_path']
            left_image_file_path = file_path_dict['left_image_file_path']
            right_image_file_path = file_path_dict['right_image_file_path']
            label_file_path = file_path_dict['label_file_path']
            
            if os.path.exists(lidar_file_path):
                lidar_point_cloud = self.read_velodyne_bin(lidar_file_path)
            else:
                print(f'Lidar {lidar_file_path} Does not Exist!')
                exit(1)
                
            if os.path.exists(calibration_file_path):
                calibration_dict = self.read_calibration_file(calibration_file_path)
            else:
                print(f'Calib {calibration_file_path} Does not Exist!')
                            
            left_image_arr = None
            right_image_arr = None
            
            if bool(left_image_file_path) and os.path.exists(left_image_file_path):
                left_image_arr = self.read_image(left_image_file_path)
                batch_data_items['image_paths'].append(left_image_file_path)
            
            if bool(right_image_file_path) and os.path.exists(right_image_file_path):
                right_image_arr = self.read_image(right_image_file_path)    
                batch_data_items['image_paths'].append(right_image_file_path)
                
            if left_image_arr is None and right_image_arr is None:
                print(f'Left Image Path {left_image_file_path} and Right Image Path {right_image_file_path}')
                exit(1)
            
            if left_image_arr is not None:
                depth_map, reflectance_map = self.preprocess(lidar_point_cloud, calibration_dict, left_image_arr)
                depth_map, reflectance_map = np.expand_dims(depth_map, axis=-1), np.expand_dims(reflectance_map, axis=-1)
                
                combined_image = np.concatenate([
                    left_image_arr, depth_map, reflectance_map
                ], axis=-1)

            elif right_image_arr is not None:
                depth_map, reflectance_map = self.preprocess(lidar_point_cloud, calibration_dict, right_image_arr)  
                depth_map, reflectance_map = np.expand_dims(depth_map, axis=-1), np.expand_dims(reflectance_map, axis=-1)
                
                combined_image = np.concatenate([
                    right_image_arr, depth_map, reflectance_map
                ], axis=-1)  
                        
            # combined_image = left_image_arr
            
            if label_file_path is not None:
                if os.path.exists(label_file_path):
                    class_labels, label_bboxes = self.read_label_file(label_file_path)                
                    transformed_dict = self.transform_sample(
                        combined_image, label_bboxes, class_labels
                    )      
                    
                    targets = self.prepare_targets(
                        idx, transformed_dict['class_labels'], transformed_dict['bboxes']
                    )                                                      
                    
                    batch_data_items['targets'].append(
                        torch.tensor(targets, dtype=torch.float32)
                    )
                    
                else:
                    print(f'Label File Path not found!!')
                    exit(1)
            
            else:
                transformed_dict = self.transform_sample(
                    combined_image
                )
                
                # transformed_dict['bboxes'] = None
                # transformed_dict['class_labels'] = None
                
            image_tensor = torch.from_numpy(transformed_dict['image']).permute((2, 0, 1))            
            batch_data_items['images'].append(image_tensor)
            
            # batch_data_items['bboxes'].append(transformed_dict['bboxes'])
            # batch_data_items['class_labels'].append(transformed_dict['class_labels'])
            
        batch_data_items['images'] = torch.stack(
            batch_data_items['images'], dim=0
        ).float()
        
        if batch_data_items['targets']:
            batch_data_items['targets'] = torch.concat(
                batch_data_items['targets'], dim=0
            )
            

        return batch_data_items
                
            