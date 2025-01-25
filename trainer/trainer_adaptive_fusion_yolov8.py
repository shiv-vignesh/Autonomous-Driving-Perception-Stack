
from tqdm import tqdm
import torch, time
import os
from collections import defaultdict
from terminaltables import AsciiTable
import numpy as np

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiLidarFusionCollateFn
from dataset_utils.enums import Enums
from model.yolov8_pointnet_fuser import Yolov8FuserPipeline

from .logger import Logger
from .trainer import Trainer


class AdaptiveFusionTrainer(Trainer):    
    def __init__(self, fuser_pipeline:Yolov8FuserPipeline, 
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
        self.logger.log_message(f'Validation Dataloader:')
        self.logger.log_new_line()
        
        self.logger.log_message(f'LiDAR Dir: {self.validation_dataloader.dataset.lidar_dir}')        
        self.logger.log_message(f'Calibration Dir: {self.validation_dataloader.dataset.calibration_dir}')
        self.logger.log_message(f'Left Image Dir: {self.validation_dataloader.dataset.left_image_dir}')
        self.logger.log_message(f'Right Image Dir: {self.validation_dataloader.dataset.right_image_dir}')
        self.logger.log_message(f'Labels Dir: {self.validation_dataloader.dataset.labels_dir}')
        self.logger.log_message(f'Train Batch Size: {self.validation_dataloader.batch_size} - Ten Percent Train Log {self.ten_percent_train_batch}')
        self.logger.log_message(f'Validation Apply Augmentation: {self.validation_dataloader.collate_fn.apply_augmentation}')
        
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
        
                 
        self.logger.log_line()
        self.logger.log_message(
            f'YOLO Device: {self.fuser_pipeline.yolo_device} - PointNet Device: {self.fuser_pipeline.point_net_device} - Adaptive Fusion Device: {self.fuser_pipeline.adaptive_fusion_device}')
        
        self.logger.log_new_line()                  

        
    def _init_dataloader(self, dataset_kwargs:dict):        
        def create_dataloader(kwargs:dict, image_resize:tuple):            
            # grid_sizes=[(20, 20), (40, 40), (80, 80)]            
            
            dataset = Kitti2DObjectDetectDataset(
                lidar_dir=kwargs['lidar_dir'],
                calibration_dir=kwargs['calibration_dir'],
                left_image_dir=kwargs['left_image_dir'],
                right_image_dir=kwargs['right_image_dir'],
                labels_dir=kwargs['labels_dir']
            )            
            
            dataloader = DataLoader(
                dataset, 
                batch_size=kwargs['batch_size'],
                collate_fn=KittiLidarFusionCollateFn(
                    image_resize=image_resize,
                    precomputed_voxel_dir=kwargs['precomputed_voxel_dir'],
                    precomputed_proj2d_dir=kwargs['precomputed_proj2d_dir'],
                    apply_augmentation=kwargs["apply_augmentation"],
                    project_2d=True,
                    voxelization=True,
                    apply_data_fusion=kwargs['apply_data_fusion'],
                    grid_sizes=[tuple(grid) for grid in kwargs['grid_sizes']]
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
                        
        if optimizer_kwargs['train_yolov8']:    
            yolov8_params = [p for p in self.fuser_pipeline.yolo.parameters() if p.requires_grad]
            
            self.yolo_optimizer = torch.optim.Adam(
                yolov8_params,
                lr=optimizer_kwargs['yolov8_lr'], 
                betas=(optimizer_kwargs['yolov8_momentum'], 0.999), 
                weight_decay=optimizer_kwargs['yolov8_decay']
            )        
            
            self.yolo_lr = optimizer_kwargs['yolov8_lr']
        
        else:
            self.yolo_optimizer = None  
            
        if optimizer_kwargs['train_pointnet']:
            pointnet_params = [p for p in self.fuser_pipeline.pointnet.parameters() if p.requires_grad]
            
            self.pointnet_optimizer = torch.optim.Adam(
                params=pointnet_params, 
                lr=optimizer_kwargs['pointnet_lr'],
                betas=(optimizer_kwargs['pointnet_momentum'], 0.999),
                weight_decay=optimizer_kwargs['pointnet_decay'], 
            )
            
        else:
            self.pointnet_optimizer = None
            
        if optimizer_kwargs['train_fusion_layers']:
            fusion_params = [p for p in self.fuser_pipeline.fusion_gates.parameters() if p.requires_grad]
            
            self.fusion_optimizer = torch.optim.AdamW(
                params=fusion_params, 
                lr=optimizer_kwargs['fusion_lr'],
                betas=(optimizer_kwargs['fusion_momentum'], 0.999),
                weight_decay=optimizer_kwargs['fusion_decay']
            )
            
        else:
            self.fusion_optimizer = None
            
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
        
        for epoch in range(1, self.epochs + 1):
            self.cur_epoch = epoch
            self.logger.log_line()
                        
            if self.monitor_train:
                self.train_one_epoch()
                            
            if self.monitor_val:
                self.valid_one_epoch()
                
            torch.cuda.empty_cache()
                
    def train_one_epoch(self):
                
        self.fuser_pipeline.train()
        
        total_loss = 0.0 
        ten_percent_batch_total_loss = 0        
        
        epoch_training_time = 0.0
        ten_percent_training_time = 0.0        
        train_iter = tqdm(self.train_dataloader, desc=f'Training Epoch: {self.cur_epoch}')
        
        self.logger.log_message(f'Training Epoch: {self.cur_epoch}')
        self.logger.log_new_line()
        
        for batch_idx, data_items in enumerate(train_iter):
            
            step_begin_time = time.time()
            outputs = self.train_one_step(data_items)
            step_end_time = time.time()            
            
            if ((batch_idx + 1) % self.gradient_accumulation_steps == 0) or (batch_idx == self.train_dataloader.__len__() - 1):
                if self.yolo_lr_burn_in:
                    self.yolo_lr = self.yolo_lr
                    batches_done = len(self.train_dataloader) * self.cur_epoch + batch_idx   
                    if batches_done < 1000: #burn-in : 1000
                        # Burn in
                        self.yolo_lr *= (batches_done / 1000)
                    else:
                        # Set and parse the learning rate to the steps defined in the cfg
                        for threshold, value in self.fuser_pipeline.yolo.hyperparams['lr_steps']:
                            if batches_done > threshold:
                                self.yolo_lr *= value
                    # Log the learning rate
                    # self.logger.log_message(f"train/learning_rate, - lr {lr} - batches_done {batches_done}")
        
                    # Set learning rate
                    for g in self.yolo_optimizer.param_groups:
                        g['lr'] = self.yolo_lr 
                        
                else:
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
                        
            total_loss += outputs['total_loss'].item()
            ten_percent_batch_total_loss += outputs['total_loss'].item()
            
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
                                                        
    def train_one_step(self, data_items):
        
        data_items['images'] = data_items['images']/255.
        
        with torch.set_grad_enabled(True):        
            outputs = self.fuser_pipeline(data_items['images'],
            data_items['raw_point_clouds'],
            data_items['proj2d_pc_mask'], 
            data_items['targets'])
            
            loss = outputs['total_loss']
        
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()           
                
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.fuser_pipeline.parameters(), self.gradient_clipping)
                
        return outputs
    
    def valid_one_epoch(self, conf_threshold = 0.25, nms_threshold = 0.5, iou_threshold=0.5):
        
        self.fuser_pipeline.eval()     
        
        val_epoch_iter = tqdm(self.validation_dataloader, desc=f'Validation Epoch: {self.cur_epoch}')
        img_size = self.fuser_pipeline.image_size
        
        total_eval_loss = 0.0
        
        self.precision = defaultdict(float)
        self.recall = defaultdict(float)
        self.f1 = defaultdict(float)
        self.ap = defaultdict(float)        
        
        for batch_idx, data_items in enumerate(val_epoch_iter):            
            with torch.no_grad():
                data_items['images'] = data_items['images']/255.
                
                outputs = self.fuser_pipeline(data_items['images'],
                data_items['raw_point_clouds'],
                data_items['proj2d_pc_mask'], 
                data_items['targets'])
                
                loss = outputs['total_loss']            
                total_eval_loss += loss.item()

                true_positives, false_positives = self.compute_metrics(outputs['feature_maps'].detach(), 
                                     data_items['targets'].to(self.fuser_pipeline.yolo_device), 
                                     conf_threshold, iou_threshold)
                
                self.compute_stats(true_positives, 
                                false_positives, 
                                len(data_items['targets']), 
                                self.fuser_pipeline.num_classes)
                                
        num_batches = len(self.validation_dataloader)
        for cls in range(self.fuser_pipeline.num_classes):
            self.ap[cls] /= num_batches
            self.precision[cls] /= num_batches
            self.recall[cls] /= num_batches
            self.f1[cls] = 2 * (self.precision[cls] * self.recall[cls]) / (self.precision[cls] + self.recall[cls] + 1e-6)
        
        self.print_eval_stats()        
                
    def compute_metrics(self, outputs:list, targets:torch.tensor, conf_threshold:float, iou_threshold=0.5):
        
        ''' 
        `outputs` is a list of two elements 
        0 - (bs, 60 + nc + 4 + 1, 8400)
            60 : additional parameters used in the Distribution Focal Loss (DFL)
            nc : num classes
            1 : objectness score 
            4 : bounding box coordinates (x, y, width, height)
        
        1 - [(bs, 60 + nc + 4 + 1, grid_h, grid_w)]
            grid = (20, 40, 80)
        
        '''
        
        # Process model outputs
        if isinstance(outputs[0], torch.Tensor) and outputs[0].dim() == 3:
            # Handle output of shape [16, 7, 8400]
            predictions = outputs[0].permute(0, 2, 1)  # [16, 8400, 7]
        else:
            # Handle output list of tensors [16, 67, 80, 80], [16, 67, 40, 40], [16, 67, 20, 20]
            predictions = []
            for output in outputs[1]:  # Assuming outputs[1] contains the list of tensors
                b, c, h, w = output.shape
                output = output.view(b, c, h*w).permute(0, 2, 1).contiguous()
                predictions.append(output)
            predictions = torch.cat(predictions, dim=1)  # [16, 8400, 67]   
            
        # Separate predictions
        box_preds = predictions[..., :4]  # (b, n, 4)
        conf_preds = predictions[..., 4].sigmoid()  # (b, n)
        cls_preds = predictions[..., 5:].sigmoid()  # (b, n, num_classes)
        
        # Apply confidence threshold
        mask = conf_preds > conf_threshold
        box_preds = box_preds[mask]
        conf_preds = conf_preds[mask]
        cls_preds = cls_preds[mask]             
        
        def xywh2xyxy(x):
            y = x.new(x.shape)
            y[..., 0] = x[..., 0] - x[..., 2] / 2
            y[..., 1] = x[..., 1] - x[..., 3] / 2
            y[..., 2] = x[..., 0] + x[..., 2] / 2
            y[..., 3] = x[..., 1] + x[..., 3] / 2
            return y
        
        # Convert predictions to xyxy format
        box_preds = xywh2xyxy(box_preds)
        
        # Process targets
        target_boxes = targets[:, 2:]
        target_labels = targets[:, 1].long()
        
        iou = box_iou(box_preds, target_boxes)

        true_positives = defaultdict(list)
        false_positives = defaultdict(list)
        
        # Match predictions to targets
        for pred_idx, target_idx in enumerate(iou.max(dim=1).indices):

            pred_class = cls_preds[pred_idx].argmax().item()
            target_class = target_labels[target_idx].item()
            
            if iou[pred_idx, target_idx] >= iou_threshold:
                pred_class = cls_preds[pred_idx].argmax().item()
                target_class = target_labels[target_idx].item()
                
                
                if pred_class == target_class:
                    true_positives[pred_class].append(conf_preds[pred_idx].cpu().item())
                else:
                    false_positives[pred_class].append(conf_preds[pred_idx].cpu().item())
            else:
                pred_class = cls_preds[pred_idx].argmax().item()
                false_positives[pred_class].append(conf_preds[pred_idx].cpu().item())
                        
        return true_positives, false_positives      
    
    def compute_stats(self, true_positives, false_positives, num_targets, num_classes):
        
        for cls in range(num_classes):
            tp = np.array(sorted(true_positives[cls], reverse=True))
            fp = np.array(sorted(false_positives[cls], reverse=True))
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
                        
            recalls = tp_cumsum / (num_targets + 1e-6)
            if len(tp_cumsum) == 0:
                precisions = np.zeros_like(fp_cumsum)
            else:
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            
            self.ap[cls] += self.compute_ap(recalls, precisions)
            
            self.precision[cls] += precisions[-1] if len(precisions) > 0 else 0
            self.recall[cls] += recalls[-1] if len(recalls) > 0 else 0
            self.f1[cls] += 2 * (self.precision[cls] * self.recall[cls]) / (self.precision[cls] + self.recall[cls] + 1e-6)
        
    def compute_ap(self, recall, precision):
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap    
    
    def print_eval_stats(self):
        
        classes = list(Enums.KiTTi_label2Id.keys())
        num_classes = len(self.precision)
        
        ap_table = [["Class", "AP", "precision", "recall", "F1"]]
        
        for i in range(num_classes):
            ap_table.append([
                classes[i], 
                f"{self.ap[i]:.4f}",
                f"{self.precision[i]:.4f}", 
                f"{self.recall[i]:.4f}", 
                f"{self.f1[i]:.4f}"                             
            ])
            
        table_string = AsciiTable(ap_table).table
        
        self.logger.log_message(f'---------- mAP per Class----------')
        self.logger.log_message(f'{table_string}')
        self.logger.log_new_line()            