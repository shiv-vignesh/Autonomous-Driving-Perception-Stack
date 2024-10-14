import torch, time
import os
from torch.utils.data import DataLoader

from .logger import Logger
from model.yolo import Darknet
from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiLidarFusionCollateFn
from trainer.loss import compute_loss

class Trainer:    
    def __init__(self, model:Darknet, 
                dataset_kwargs:dict, optimizer_kwargs:dict,
                trainer_kwargs:dict, lr_scheduler_kwargs:dict):
                
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
        
        self._init_dataloader(dataset_kwargs)
        self._init_optimizer(optimizer_kwargs)
        
        if lr_scheduler_kwargs:
            self._init_lr_scheduler(lr_scheduler_kwargs)
            
        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 10             
        
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
                batch_size=self.model.hyperparams['batch']//self.model.hyperparams['subdivisions'],
                collate_fn=KittiLidarFusionCollateFn(
                    image_resize=image_resize,
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

        for epoch in range(self.epochs):
            self.cur_epoch = epoch
            self.logger.log_line()
            
            if self.monitor_train:
                self.train_one_epoch()
                
            if self.monitor_val and self.validation_dataloader is not None:
                self.valid_one_epoch()
        
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
        
        for batch_idx, data_items in enumerate(self.train_dataloader):
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
    
    def valid_one_epoch(self):
        
        pass