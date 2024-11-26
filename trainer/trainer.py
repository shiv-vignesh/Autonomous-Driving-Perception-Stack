from tqdm import tqdm
import torch, time
import os, math
import numpy as np
from torch.utils.data import DataLoader
from terminaltables import AsciiTable

from .logger import Logger
from model.yolo import Darknet
from model.yolo_utils import xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class
from CSCI739.dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiLidarFusionCollateFn
from dataset_utils.enums import Enums
from trainer.loss import compute_loss

class Trainer:    
    def __init__(self, model:Darknet, 
                dataset_kwargs:dict, optimizer_kwargs:dict,
                trainer_kwargs:dict, lr_scheduler_kwargs:dict):
        
        ''' 
        TODO
        1. Add Logs after _init_dataloader, _init_optimizer 
        2. Implement Callbacks for model checkpointing 
        3. Complete Validation methods 
        '''
                
        self.model = model 
        
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
        
        self.device_count = torch.cuda.device_count()         
        
        self.device = torch.device(trainer_kwargs["device"]) if torch.cuda.is_available() else torch.device("cpu")        
        self.model.to(self.device)        
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)        

        self.logger = Logger(trainer_kwargs)     
        
        '''
        TODO, load model from ckpt
        '''   
        
        self.model.to(self.device)
        self.batch_size = self.model.hyperparams['batch']//self.model.hyperparams['subdivisions']              
        
        self._init_dataloader(dataset_kwargs)
        
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
        self.logger.log_message(f'Train Batch Size: {self.validation_dataloader.batch_size}')
        self.logger.log_message(f'Validation Apply Augmentation: {self.validation_dataloader.collate_fn.apply_augmentation}')
        
        self.logger.log_line()        
        
        self._init_optimizer(optimizer_kwargs)
        self.logger.log_line()
        self.logger.log_message(f'Optimizer: {self.optimizer.__class__.__name__}')
        self.logger.log_new_line()
        
        if lr_scheduler_kwargs:
            self._init_lr_scheduler(lr_scheduler_kwargs)
            
        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 10     
        
        self.logger.log_line()
        self.logger.log_message(f'Device: {self.device} and Device Count: {self.device_count}')
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
                    apply_augmentation=kwargs["apply_augmentation"]
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
        
        params = [p for p in self.model.parameters() if p.requires_grad]        
        if optimizer_kwargs['type'] == "AdamW":
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.model.hyperparams['learning_rate'],
                weight_decay=self.model.hyperparams['decay']
            )
            
        elif optimizer_kwargs['type'] == "SGD":
            self.optimizer = torch.optim.SGD(
                params,
                lr=self.model.hyperparams['learning_rate'],
                weight_decay=self.model.hyperparams['decay'],
                momentum=self.model.hyperparams['momentum']
            )
            
        else:
            self.logger.log_message(
                f"Unknowm Optimizer: {optimizer_kwargs['type']}. Choose Between AdamW and SGD"
            )
            self.logger.log_new_line()
            exit(1)
            
    def _init_lr_scheduler(self, lr_scheduler_kwargs:dict):        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                            step_size=lr_scheduler_kwargs['step_size'], 
                                                            gamma=lr_scheduler_kwargs['gamma'])
        
    def train(self):
        
        self.logger.log_line()
        self.logger.log_message(
            f'Training: Max Epoch - {self.epochs}'
        )
        self.logger.log_new_line()
        
        self.total_training_time = 0.0
        
        self.cur_epoch = 0        
        
        # self.valid_one_epoch()
        # exit(1)
                
        for epoch in range(self.epochs):
            self.cur_epoch = epoch
            self.logger.log_line()
            
            if self.monitor_train:
                self.train_one_epoch()
                
                if (self.cur_epoch + 1) % self.checkpoint_idx == 0:
                    torch.save(self.model.state_dict(), f'{self.output_dir}/yolo_weights_{self.cur_epoch}.pth')
                
            if self.monitor_val and self.validation_dataloader is not None:
                self.valid_one_epoch()
            
            # exit(1)
        
        # try:
        #     for epoch in range(self.epochs):
        #         self.cur_epoch = epoch
        #         self.logger.log_line()
                
        #         if self.monitor_train:
        #             self.train_one_epoch()
                    
        #         if self.monitor_val and self.validation_dataloader is not None:
        #             self.valid_one_epoch()
                    
        # except:
        #     self.logger.log_line()
        #     self.logger.log_message(f'Exiting Training due to Keyboard Interrupt')
        #     exit(1)            
            
    def train_one_epoch(self):
        
        self.model.train()
        
        total_loss = 0.0 
        ten_percent_batch_total_loss = 0
        
        epoch_training_time = 0.0
        ten_percent_training_time = 0.0        
        
        train_iter = tqdm(self.train_dataloader)
        
        for batch_idx, data_items in enumerate(train_iter):
            for k,v in data_items.items():
                if torch.is_tensor(v):                    
                    data_items[k] = v.to(self.device)
                    
            step_begin_time = time.time()
            loss, loss_components = self.train_one_step(data_items)
            step_end_time = time.time()
                        
            
            batches_done = len(self.train_dataloader) * self.cur_epoch + batch_idx   
            if batches_done % self.model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = self.model.hyperparams['learning_rate']
                if batches_done < self.model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / self.model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in self.model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                # self.logger.log_message(f"train/learning_rate, - lr {lr} - batches_done {batches_done}")
    
                # Set learning rate
                for g in self.optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                self.optimizer.step()
                # Reset gradients
                self.optimizer.zero_grad()                
                
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
                                      
                    
    def train_one_step(self, data_items:dict):    
        with torch.set_grad_enabled(True):  
            outputs = self.model(data_items['images'])
            loss, loss_components = compute_loss(outputs, data_items['targets'], self.model)                        
            loss.backward()

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

        self.model.eval()
        
        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        
        val_epoch_iter = tqdm(self.validation_dataloader, disable=True)   

        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        img_size = self.model.hyperparams['height']
        
        total_eval_loss = 0.0
        
        for batch_idx, data_items in enumerate(val_epoch_iter):
            for k,v in data_items.items():
                if torch.is_tensor(v):                    
                    data_items[k] = v.to(self.device)
                    
            # size --> [bs*num_labels_per_batch, 6]
            #6 --> [batch_idx, class_id, x_center, y_center, width, height]
            
            targets = data_items['targets'].cpu()
            labels += targets[:, 1] #[class_id] 
                        
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size
            
            with torch.no_grad():
                outputs = self.model(data_items['images'])
                
                #converting from [bs, grid_size_flat, num_classes] to [bs, num_anchors, grid, grid, num_classes]                
                # in-place operation on outputs (Reshaping)
                reshaped_outputs = reshape_outputs(outputs)
                loss, loss_components = compute_loss(reshaped_outputs, data_items['targets'], self.model, eval_debug=True)
                total_eval_loss += loss.item()
                                
                del reshaped_outputs
                                            
                anchor_grids = [yolo_layer.anchor_grid for yolo_layer in self.model.yolo_layers]            
                outputs = apply_sigmoid_activation(outputs, data_items['images'].size(2), anchor_grids)                
                outputs = non_max_suppression(outputs)
            
            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=0.45)
        
        self.logger.log_new_line()
        self.logger.log_message(f'Epoch {self.cur_epoch} - Evaluation Loss {total_eval_loss/len(self.validation_dataloader):.4f}')    
        self.logger.log_line()
        
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*sample_metrics))]            

        metrics_output = ap_per_class(
            true_positives, pred_scores, pred_labels, labels) 
        
        self.print_eval_stats(metrics_output, list(Enums.KiTTi_label2Id.keys()), True)
        
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
 