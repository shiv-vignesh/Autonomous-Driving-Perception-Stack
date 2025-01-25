import torch.utils
from tqdm import tqdm
import torch, time
import os, math, random, cv2
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from terminaltables import AsciiTable
from scipy.interpolate import griddata

from .logger import Logger
from model.yolov3_pointnet_fuser import FuserPipeline
from model.yolo_utils import xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class
from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiLidarFusionCollateFn
from dataset_utils.enums import Enums
from trainer.loss import compute_loss, feature_alignment_loss

from .trainer import Trainer

class AugmentImage:
    
    def __init__(self, fog_intensity=0.65, salt_prob=0.1, pepper_prob=0.1, pixel_size:int=8):
        """
        Atmospheric fog augmentation class using depth and reflectance maps.
        :param beta_range: Range of fog density (scattering coefficient).
        :param airlight_intensity: Range for airlight brightness (fog background light).
        """

        self.fog_intensity = fog_intensity
        self.alpha = 1 - self.fog_intensity
        self.beta = self.fog_intensity 
        self.gamma = 0.0
        
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        self.pixel_size = pixel_size
                
    def apply_haze(self, images:torch.tensor):
        
        orig_range = (images.min(), images.max())
        if orig_range[1] > 1:  # Assume range is [0, 255]
            images = images / 255.0                
            
        depth_map = images[:, -2, :, :] # RGB + depth + reflectance
        reflectance = images[:, -1, :, :]
        rgb = images[:, :3, :, :]
                
        haze_layer = torch.ones_like(rgb, device=rgb.device, dtype=rgb.dtype) * self.fog_intensity
        foggy_images = rgb * self.alpha + (1-haze_layer)*self.beta + self.gamma

        if orig_range[1] > 1:
            foggy_images = (foggy_images * 255).to(torch.float)  

        # foggy_images = foggy_images.detach().cpu().numpy()
        # for idx, foggy_image in enumerate(foggy_images):
        #     foggy_image = np.transpose(foggy_image, (1, 2, 0))
        #     cv2.imwrite(f'{idx}_foggy.png', foggy_image)        
        
        # exit(1)

        foggy_images = torch.concat([foggy_images, depth_map.unsqueeze(1), reflectance.unsqueeze(1)], dim=1)
                              
        return foggy_images
    
    def apply_salt_pepper(self, images:torch.tensor):

        orig_range = (images.min(), images.max())
        if orig_range[1] > 1:  # Assume range is [0, 255]
            images = images / 255.0                
            
        depth_map = images[:, -2, :, :] # RGB + depth + reflectance
        reflectance = images[:, -1, :, :]
        rgb = images[:, :3, :, :]

        # Create a tensor for salt noise
        salt_mask = (torch.rand_like(rgb) < self.salt_prob).float()

        # Create a tensor for pepper noise
        pepper_mask = (torch.rand_like(rgb) < self.pepper_prob).float()

        # Add salt noise (set pixels to 1)
        images_with_salt = rgb + salt_mask

        # Add pepper noise (set pixels to 0)
        images_with_pepper = images_with_salt - pepper_mask

        # Clamp values to ensure they stay within the [0, 1] range
        noisy_images = torch.clamp(images_with_pepper, 0.0, 1.0)        
        
        if orig_range[1] > 1:
            noisy_images = (noisy_images * 255).to(torch.float)  

        noisy_images_2 = noisy_images.detach().cpu().numpy()
        for idx, foggy_image in enumerate(noisy_images_2):
            foggy_image = np.transpose(foggy_image, (1, 2, 0))
            
            cv2.imwrite(f'{idx}_saltpepper.png', foggy_image)    
            
        noisy_images = torch.concat([noisy_images, depth_map.unsqueeze(1), reflectance.unsqueeze(1)], dim=1)
            
        return noisy_images
    
    def pixelate(self, images:torch.tensor):
        
        orig_range = (images.min(), images.max())
        if orig_range[1] > 1:  # Assume range is [0, 255]
            images = images / 255.0                
            
        depth_map = images[:, -2, :, :] # RGB + depth + reflectance
        reflectance = images[:, -1, :, :]
        rgb = images[:, :3, :, :]        

        batch_size, channels, height, width = rgb.shape

        # Compute the new height and width after downsampling
        new_height = height // self.pixel_size
        new_width = width // self.pixel_size

        # Downsample the image to the new size using average pooling
        rgb_resized = F.avg_pool2d(rgb, kernel_size=self.pixel_size, stride=self.pixel_size)

        # Upscale the image back to the original size using nearest neighbor interpolation
        pixelated_images = F.interpolate(rgb_resized, size=(height, width), mode='nearest')

        if orig_range[1] > 1:
            pixelated_images = (pixelated_images * 255).to(torch.float)  
            
        pixelated_images_2 = pixelated_images.detach().cpu().numpy()
        for idx, foggy_image in enumerate(pixelated_images_2):
            foggy_image = np.transpose(foggy_image, (1, 2, 0))
            
            cv2.imwrite(f'{idx}_pixelate.png', foggy_image)                
            
        pixelated_images = torch.concat([pixelated_images, depth_map.unsqueeze(1), reflectance.unsqueeze(1)], dim=1)

        return pixelated_images        
    
    def __call__(self, images:torch.tensor, augmentation:str):
        
        if 'transmittance' == augmentation:
            return self.apply_haze(images)

        elif 'SaltPapperNoise' == augmentation:
            return self.apply_salt_pepper(images)  
        
        elif 'pixelate' == augmentation:
            return self.pixelate(images)            

class AdaptiveFusionTrainer(Trainer):    
    def __init__(self, fuser_pipeline:FuserPipeline, 
                dataset_kwargs:dict, optimizer_kwargs:dict,
                trainer_kwargs:dict, lr_scheduler_kwargs:dict):
        
        self.fuser_pipeline = fuser_pipeline 
        
        self.output_dir = trainer_kwargs['output_dir']
        self.is_training = trainer_kwargs["is_training"]
        self.first_val_epoch = trainer_kwargs["first_val_epoch"]
        self.metric_eval_mode = trainer_kwargs["metric_eval_mode"]
        self.metric_average_mode = trainer_kwargs["metric_average_mode"]
        self.epochs = trainer_kwargs["epochs"]
        self.monitor_train = trainer_kwargs["monitor_train"]
        self.monitor_val = trainer_kwargs["monitor_val"]
        self.gradient_clipping = trainer_kwargs["gradient_clipping"]
        
        self.checkpoint_idx = trainer_kwargs['checkpoint_idx']     
        self.robustness_augmentations = trainer_kwargs['robustness_augmentations']
        self.gradient_accumulation_steps = trainer_kwargs['gradient_accumulation_steps']
        self.compute_feature_alignment = trainer_kwargs['compute_feature_alignment']
        self.yolo_lr_burn_in = trainer_kwargs['yolo_lr_burn_in']

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)                
        
        self.logger = Logger(trainer_kwargs) 
        
        self.batch_size = self.fuser_pipeline.yolo.hyperparams['batch']//self.fuser_pipeline.yolo.hyperparams['subdivisions']              
        
        self._init_dataloader(dataset_kwargs)
        
        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 10          
        
        self.logger.log_line()
        self.logger.log_message(f'Train Dataloader:')
        self.logger.log_new_line()
        
        self.logger.log_message(f'LiDAR Dir: {self.train_dataloader.dataset.lidar_dir}')        
        self.logger.log_message(f'Calibration Dir: {self.train_dataloader.dataset.calibration_dir}')
        self.logger.log_message(f'Left Image Dir: {self.train_dataloader.dataset.left_image_dir}')
        self.logger.log_message(f'Right Image Dir: {self.train_dataloader.dataset.right_image_dir}')
        self.logger.log_message(f'Labels Dir: {self.train_dataloader.dataset.labels_dir}')
        self.logger.log_message(f'Train Batch Size: {self.train_dataloader.batch_size}')
        self.logger.log_message(f'Train Apply Augmentation: {self.train_dataloader.collate_fn.apply_augmentation}')
        
        self.logger.log_line()
        
        self.logger.log_line()
        self.logger.log_message(f'Validation Dataloader:')
        self.logger.log_new_line()
        
        self.logger.log_message(f'LiDAR Dir: {self.validation_dataloader.dataset.lidar_dir}')        
        self.logger.log_message(f'Calibration Dir: {self.validation_dataloader.dataset.calibration_dir}')
        self.logger.log_message(f'Left Image Dir: {self.validation_dataloader.dataset.left_image_dir}')
        self.logger.log_message(f'Right Image Dir: {self.validation_dataloader.dataset.right_image_dir}')
        self.logger.log_message(f'Labels Dir: {self.validation_dataloader.dataset.labels_dir}')
        self.logger.log_message(f'Train Batch Size: {self.validation_dataloader.batch_size} - Ten Percent Train Log {self.ten_percent_train_batch}')
        self.logger.log_message(f'Validation Apply Augmentation: {self.validation_dataloader.collate_fn.apply_augmentation}')
        
        self.logger.log_new_line()        
        
        if self.robustness_augmentations:
            self.logger.log_message(f'Robustness Augmentations: {self.robustness_augmentations}')
            self.robust_augmentor = AugmentImage()
        
        self.logger.log_line()        
        
        self._init_optimizer(
            optimizer_kwargs
        )
        
        self._init_lr_scheduler(lr_scheduler_kwargs)
                
        if self.yolo_optimizer is not None:            
            self.logger.log_message(f"Group {self.fuser_pipeline.yolo._get_name()}: {self.yolo_optimizer.__class__.__name__}")
            self.logger.log_message(f"  Learning Rate: {self.yolo_optimizer.param_groups[0]['lr']}")
            self.logger.log_message(f"  Weight Decay: {self.yolo_optimizer.param_groups[0]['weight_decay']}")
            self.logger.log_message(f"  Number of Parameters: {sum([len(param['params']) for param in self.yolo_optimizer.param_groups])}")  
            self.logger.log_new_line()    
                    
        if self.pointnet_optimizer is not None:
            self.logger.log_message(f"Group {self.fuser_pipeline.pointnet._get_name()}: {self.pointnet_optimizer.__class__.__name__}")
            self.logger.log_message(f"  Learning Rate: {self.pointnet_optimizer.param_groups[0]['lr']}")
            self.logger.log_message(f"  Weight Decay: {self.pointnet_optimizer.param_groups[0]['weight_decay']}")
            self.logger.log_message(f"  Number of Parameters: {sum([len(param['params']) for param in self.pointnet_optimizer.param_groups])}")  
            self.logger.log_new_line()                          
            
        if self.fusion_optimizer is not None:
            self.logger.log_message(f"Group {self.fuser_pipeline.fusion_gates._get_name()}: {self.fusion_optimizer.__class__.__name__}")
            self.logger.log_message(f"  Learning Rate: {self.fusion_optimizer.param_groups[0]['lr']}")
            self.logger.log_message(f"  Weight Decay: {self.fusion_optimizer.param_groups[0]['weight_decay']}")
            self.logger.log_message(f"  Number of Parameters: {sum([len(param['params']) for param in self.fusion_optimizer.param_groups])}")  
            self.logger.log_new_line()
        
        if lr_scheduler_kwargs:
            self._init_lr_scheduler(lr_scheduler_kwargs)            
        
        self.logger.log_line()
        self.logger.log_message(
            f'YOLO Device: {self.fuser_pipeline.yolo_device} - PointNet Device: {self.fuser_pipeline.point_net_device} - Adaptive Fusion Device: {self.fuser_pipeline.adaptive_fusion_device}')
        
        self.logger.log_new_line()                  

    def _init_dataloader(self, dataset_kwargs:dict):        
        def create_dataloader(kwargs:dict, image_resize:tuple):
            dataset = Kitti2DObjectDetectDataset(
                lidar_dir=kwargs['lidar_dir'],
                calibration_dir=kwargs['calibration_dir'],
                left_image_dir=kwargs['left_image_dir'],
                right_image_dir=kwargs['right_image_dir'],
                labels_dir=kwargs['labels_dir']
            )            
            dataloader = DataLoader(
                dataset, 
                # batch_size=kwargs['batch_size'],
                batch_size=self.batch_size,
                collate_fn=KittiLidarFusionCollateFn(
                    image_resize=image_resize,
                    precomputed_voxel_dir=kwargs['precomputed_voxel_dir'],
                    precomputed_proj2d_dir=kwargs['precomputed_proj2d_dir'],
                    apply_augmentation=kwargs["apply_augmentation"],
                    project_2d=True,
                    voxelization=True
                ),
                shuffle=kwargs['shuffle']
            )            
            return dataloader
        
        if dataset_kwargs['trainer_dataset_kwargs']:
            self.train_dataloader = create_dataloader(
                dataset_kwargs['trainer_dataset_kwargs'], dataset_kwargs['image_resize']
            )        
        else:
            self.logger.log_line()
            self.logger.log_message(
                f'Trainer Kwargs not Found: {dataset_kwargs["trainer_kwargs"]}'
            )
            exit(1)
        
        if dataset_kwargs['validation_dataset_kwargs']:
            self.validation_dataloader = create_dataloader(
                dataset_kwargs['validation_dataset_kwargs'], dataset_kwargs['image_resize']
            )        
        else:
            self.validation_dataloader = None
                         
    def _init_optimizer(self, optimizer_kwargs:dict):
                
        yolo_params_dict = []
        
        if optimizer_kwargs['train_yolo_backbone']:    
            backbone_params = self.fuser_pipeline.yolo.get_backbone_trainable_params(
                requires_grad=optimizer_kwargs['train_yolo_backbone'])
                        
            yolo_params_dict.append({
                'params':backbone_params,
                'lr':self.fuser_pipeline.yolo.hyperparams['learning_rate'],
                'weight_decay':self.fuser_pipeline.yolo.hyperparams['decay'],
                'name':'yolo_backbone'
            })

        if optimizer_kwargs['train_yolo_detection']:    
            detection_head_params = self.fuser_pipeline.yolo.get_detection_head_params(
                requires_grad=optimizer_kwargs['train_yolo_detection'])
                            
            yolo_params_dict.append({
                'params':detection_head_params,
                'lr':self.fuser_pipeline.yolo.hyperparams['learning_rate'],
                'weight_decay':self.fuser_pipeline.yolo.hyperparams['decay'],
                'name':'yolo_detection_head'
            })   
            
        if yolo_params_dict:
            self.yolo_optimizer = torch.optim.Adam(
                yolo_params_dict, 
                lr=1e-3, 
                weight_decay=1e-4
            )        
        
        else:
            self.yolo_optimizer = None  
            
        if optimizer_kwargs['train_pointnet']:
            pointnet_params = [p for p in self.fuser_pipeline.pointnet.parameters() if p.requires_grad]
            
            self.pointnet_optimizer = torch.optim.Adam(
                params=pointnet_params, 
                lr=optimizer_kwargs['pointnet_lr'],
                weight_decay=optimizer_kwargs['pointnet_decay'], 
            )
            
        else:
            self.pointnet_optimizer = None
            
        if optimizer_kwargs['train_fusion_layers']:
            fusion_params = [p for p in self.fuser_pipeline.fusion_gates.parameters() if p.requires_grad]
            
            self.fusion_optimizer = torch.optim.AdamW(
                params=fusion_params, 
                lr=optimizer_kwargs['fusion_lr'],
                weight_decay=optimizer_kwargs['fusion_decay']
            )
            
        else:
            self.fusion_optimizer = None
                
    def _init_optimizer_2(self, optimizer_kwargs:dict):
        
        params_dict = []

        backbone_params = self.fuser_pipeline.yolo.get_backbone_trainable_params(
            requires_grad=optimizer_kwargs['train_yolo_backbone'])
        
        detection_head_params = self.fuser_pipeline.yolo.get_detection_head_params(
            requires_grad=optimizer_kwargs['train_yolo_detection'])
    
        if optimizer_kwargs['train_yolo_backbone']:    
            params_dict.append({
                'params':backbone_params,
                'lr':self.fuser_pipeline.yolo.hyperparams['learning_rate'],
                'weight_decay':self.fuser_pipeline.yolo.hyperparams['decay'],
                'name':'yolo_backbone'
            })
                            
        if optimizer_kwargs['train_yolo_detection']:        
            params_dict.append({
                'params':detection_head_params,
                'lr':self.fuser_pipeline.yolo.hyperparams['learning_rate'],
                'weight_decay':self.fuser_pipeline.yolo.hyperparams['decay'],
                'name':'yolo_detection_head'
            })                        
            
        if optimizer_kwargs['train_pointnet']:
            params_dict.append({
                'params':[p for p in self.fuser_pipeline.pointnet.parameters() if p.requires_grad],
                'lr':optimizer_kwargs['pointnet_lr'],
                'weight_decay':optimizer_kwargs['pointnet_decay'],
                "name":'pointnet'
            })
            
        if optimizer_kwargs['train_fusion_layers']:
            params_dict.append({
                'params':[p for p in self.fuser_pipeline.fusion_gates.parameters() if p.requires_grad],
                'lr':optimizer_kwargs['fusion_lr'],
                'weight_decay':optimizer_kwargs['fusion_decay'],
                "name":'fusion_layers'
            })            
        
        if optimizer_kwargs['type'] == "AdamW":
            self.optimizer = torch.optim.Adam(
                params_dict
            )
            
        elif optimizer_kwargs['type'] == "SGD":
            self.optimizer = torch.optim.SGD(
                params_dict
            )
            
        else:
            self.logger.log_message(
                f"Unknowm Optimizer: {optimizer_kwargs['type']}. Choose Between AdamW and SGD"
            )
            self.logger.log_new_line()
            exit(1)                          
    
    def _init_lr_scheduler(self, lr_scheduler_kwargs):
        # return super()._init_lr_scheduler(lr_scheduler_kwargs)
        
        if self.yolo_optimizer is not None:            
            self.yolo_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.yolo_optimizer, T_max=50
            )            
        else:
            self.yolo_lr_scheduler = None

        if self.pointnet_optimizer is not None:            
            self.pointnet_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.pointnet_optimizer, T_max=50
            )            
        else:
            self.pointnet_lr_scheduler = None
            
        if self.fusion_optimizer is not None:            
            self.fusion_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.fusion_optimizer, T_max=50
            )                        
        else:
            self.fusion_lr_scheduler = None
            
    def train(self):
        
        self.logger.log_line()
        self.logger.log_message(
            f'Training: Max Epoch - {self.epochs}'
        )
        self.logger.log_new_line()
        
        self.total_training_time = 0.0
        
        self.cur_epoch = 0   
        self.best_score = 0.0     
        
        # self.valid_one_epoch()
        # exit(1)
                
        for epoch in range(self.epochs):
            self.cur_epoch = epoch
            self.logger.log_line()
            
            if self.monitor_train:
                self.train_one_epoch()
                
                if (self.cur_epoch + 1) % self.checkpoint_idx == 0:
                    ckpt_dir = f'{self.output_dir}/ckpt_{self.cur_epoch}'
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    
                    self.fuser_pipeline.save_model_ckpts(
                        ckpt_dir, self.cur_epoch
                    )
                
            if self.monitor_val and self.validation_dataloader is not None:
                self.valid_one_epoch()
            
            # exit(1)
        
    def train_one_epoch(self):
        
        self.fuser_pipeline.train()
        
        total_loss = 0.0 
        ten_percent_batch_total_loss = 0
        
        epoch_training_time = 0.0
        ten_percent_training_time = 0.0        
        train_iter = tqdm(self.train_dataloader, desc=f'Training Epoch: {self.cur_epoch}')
        for batch_idx, data_items in enumerate(train_iter):
                    
            step_begin_time = time.time()
            
            if self.robustness_augmentations:            
                loss, loss_components = self.train_one_step_augmentation(data_items)       
            else:
                loss, loss_components = self.train_one_step(data_items)       
            step_end_time = time.time()
            
            if ((batch_idx + 1) % self.gradient_accumulation_steps == 0) or (batch_idx == self.train_dataloader.__len__() - 1):   
                     
                if self.cur_epoch > 20 and self.yolo_lr_burn_in:                
                    lr = self.fuser_pipeline.yolo.hyperparams['learning_rate']
                    batches_done = len(self.train_dataloader) * self.cur_epoch + batch_idx   
                    if batches_done < self.fuser_pipeline.yolo.hyperparams['burn_in']:
                        # Burn in
                        lr *= (batches_done / self.fuser_pipeline.yolo.hyperparams['burn_in'])
                    else:
                        # Set and parse the learning rate to the steps defined in the cfg
                        for threshold, value in self.fuser_pipeline.yolo.hyperparams['lr_steps']:
                            if batches_done > threshold:
                                lr *= value
                    # Log the learning rate
                    # self.logger.log_message(f"train/learning_rate, - lr {lr} - batches_done {batches_done}")
        
                    # Set learning rate
                    for g in self.yolo_optimizer.param_groups:
                        g['lr'] = lr 
                        
                if self.yolo_optimizer is not None:                
                    self.yolo_optimizer.step()
                    self.yolo_lr_scheduler.step()
                    self.yolo_optimizer.zero_grad()  
                    
                if self.pointnet_optimizer is not None:                
                    self.pointnet_optimizer.step()
                    self.pointnet_lr_scheduler.step()
                    self.pointnet_optimizer.zero_grad()  
                    
                if self.fusion_optimizer is not None:                
                    self.fusion_optimizer.step()
                    self.fusion_lr_scheduler.step()
                    self.fusion_optimizer.zero_grad()                                                                                   
                                                    
            total_loss += loss.item()
            ten_percent_batch_total_loss += loss.item()
            
            epoch_training_time += (step_end_time - step_begin_time)
            ten_percent_training_time += (step_end_time - step_begin_time)            

            if (batch_idx + 1) % self.ten_percent_train_batch == 0:
                average_loss = ten_percent_batch_total_loss/self.ten_percent_train_batch
                average_time = ten_percent_training_time/self.ten_percent_train_batch    
                
                message = f'Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - total loss {average_loss:.4f}'
                self.logger.log_message(message=message)
                
                ten_percent_batch_total_loss = 0
                ten_percent_training_time = 0.0   
                
        self.logger.log_message(
            f'Epoch {self.cur_epoch} - Average Loss {total_loss/self.total_train_batch:.4f}'
        )
    
    def train_one_step_augmentation(self, data_items:dict):
        
        pixelated_images = self.robust_augmentor(data_items['images'], 'pixelate')            
        salt_pepper_images = self.robust_augmentor(data_items['images'], 'SaltPapperNoise')
        
        exit(1)
        
        with torch.set_grad_enabled(True):
            outputs, yolo_features_list, lidar_features_list = self.fuser_pipeline(
                data_items['images'],
                data_items['raw_point_clouds'],
                data_items['proj2d_pc_mask'])            
                        
            clean_loss, loss_components = compute_loss(outputs, 
                                                data_items['targets'].to(self.fuser_pipeline.yolo_device), 
                                                self.fuser_pipeline.yolo)              
                
            with torch.autograd.set_detect_anomaly(True):
                clean_loss.backward()
            
            if random.random() > 0.5:
                data_items['images'] = salt_pepper_images                
            else:
                data_items['images'] = pixelated_images                
            
            outputs, yolo_features_list, lidar_features_list = self.fuser_pipeline(
                data_items['images'],
                data_items['raw_point_clouds'],
                data_items['proj2d_pc_mask']) 
            
            disturbed_loss, _ = compute_loss(outputs, 
                                            data_items['targets'].to(self.fuser_pipeline.yolo_device), 
                                            self.fuser_pipeline.yolo)   
                
            with torch.autograd.set_detect_anomaly(True):
                disturbed_loss.backward()                                         
        
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.fuser_pipeline.parameters(), self.gradient_clipping)
        
        return clean_loss, loss_components   
        
    def train_one_step(self, data_items:dict):   

        with torch.set_grad_enabled(True):  
            outputs, yolo_features_list, lidar_features_list = self.fuser_pipeline(
                data_items['images'],
                data_items['raw_point_clouds'],
                data_items['proj2d_pc_mask'])            
                        
            loss, loss_components = compute_loss(outputs, 
                                                data_items['targets'].to(self.fuser_pipeline.yolo_device), 
                                                self.fuser_pipeline.yolo)                       
                                                 
            if self.compute_feature_alignment:            
                total_align_loss = torch.zeros(1, device=loss.device)
                for yolo_feature, lidar_feature in zip(yolo_features_list, lidar_features_list):
                    align_loss = feature_alignment_loss(yolo_feature.to(loss.device), lidar_feature.to(loss.device))
                    total_align_loss += align_loss * 0.1            
                
                with torch.autograd.set_detect_anomaly(True):
                    total_align_loss.backward(retain_graph=True)
                    
                loss += total_align_loss
                
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()           
                
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.fuser_pipeline.parameters(), self.gradient_clipping)                 

        return loss, loss_components      
    
    def valid_one_epoch(self, conf_threshold:float=0.4, nms_threshold:float=0.5):
        
        def reshape_outputs(outputs:list):
            num_anchors = 3
            
            for i, x in enumerate(outputs):
                bs, num_preds, _ = x.shape
                grid_size = int(math.sqrt(num_preds // num_anchors))
                outputs[i] = x.view(bs, num_anchors, grid_size, grid_size, -1)
                
            return outputs
        
        def make_grid(nx, ny, device):
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid = torch.stack((xv, yv), 2).float().to(device)
            return grid        
        
        def apply_sigmoid_activation(outputs:list, img_size, anchor_grids):
                        
            for i,(x, anchor_grid) in enumerate(zip(outputs, anchor_grids)):         
                bs, num_anchors, grid_size_y, grid_size_x, num_classes = x.shape
                stride = img_size // x.size(2)
                
                grid = make_grid(grid_size_x, grid_size_y, x.device)
                
                x[..., 0:2] = (x[..., 0:2].sigmoid() + grid) * stride  # xy
                x[..., 2:4] = torch.exp(x[..., 2:4]) * anchor_grid # wh
                x[..., 4:] = x[..., 4:].sigmoid() # objectness_score, classes      
                                    
                outputs[i] = x.view(bs, -1, num_classes) # number of outputs per anchor
                
            return torch.cat(outputs, 1)

        self.fuser_pipeline.eval()
        
        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        
        val_epoch_iter = tqdm(self.validation_dataloader, disable=True)   

        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        img_size = self.fuser_pipeline.yolo.hyperparams['height']
        
        total_eval_loss = 0.0
        
        for batch_idx, data_items in enumerate(val_epoch_iter):

            # size --> [bs*num_labels_per_batch, 6]
            #6 --> [batch_idx, class_id, x_center, y_center, width, height]
            
            targets = data_items['targets'].cpu()
            labels += targets[:, 1] #[class_id] 
                        
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size
            
            with torch.no_grad():
                outputs, yolo_features_list, lidar_features_list = self.fuser_pipeline(
                    data_items['images'],
                    data_items['raw_point_clouds'],
                    data_items['proj2d_pc_mask'])                    
                
                #converting from [bs, grid_size_flat, num_classes] to [bs, num_anchors, grid, grid, num_classes]                
                # in-place operation on outputs (Reshaping)
                reshaped_outputs = reshape_outputs(outputs)
                loss, loss_components = compute_loss(reshaped_outputs,
                                                data_items['targets'].to(self.fuser_pipeline.yolo_device), 
                                                self.fuser_pipeline.yolo, eval_debug=True)
                total_eval_loss += loss.item()
                                
                del reshaped_outputs
                                            
                anchor_grids = [yolo_layer.anchor_grid for yolo_layer in self.fuser_pipeline.yolo.yolo_layers]            
                outputs = apply_sigmoid_activation(outputs, data_items['images'].size(2), anchor_grids)                
                outputs = non_max_suppression(outputs)
            
            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=0.5)
        
        self.logger.log_new_line()
        self.logger.log_message(f'Epoch {self.cur_epoch} - Evaluation Loss {total_eval_loss/len(self.validation_dataloader):.4f}')    
        self.logger.log_line()
        
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*sample_metrics))]            

        metrics_output = ap_per_class(
            true_positives, pred_scores, pred_labels, labels) 
        
        self.print_eval_stats(metrics_output, list(Enums.KiTTi_label2Id.keys()), True)
        
        _, _, AP, _, _ = metrics_output
        
        if AP.mean() > (self.best_score + 0.02):
            self.best_score = AP.mean()
            ckpt_dir = f'{self.output_dir}/best-model'
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            
            self.fuser_pipeline.save_model_ckpts(
                ckpt_dir, self.cur_epoch
            )            
            
            self.logger.log_message(f'Saving Best Model at Performance - AP: {self.best_score}')
            self.logger.log_line()
        
    def print_eval_stats(self, metrics_output, class_names, verbose):
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
                        f'{AP[i]}:.5f',
                        f'{precision[i]:.5f}',
                        f'{recall[i]:.5f}',
                        f'{f1[i]:.5f}'
                    ])
                
                table_string = AsciiTable(ap_table).table
                
                self.logger.log_message(f'---------- mAP per Class----------')
                self.logger.log_message(f'{table_string}')
                self.logger.log_new_line()
                self.logger.log_message(f'---------- Total mAP {AP.mean():.5f} ----------')
                
        else:
            self.logger.log_message("---- mAP not measured (no detections found by model) ----")                               