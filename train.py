import json, os
import torch
import numpy as np
from model.yolo import Darknet
from model.yolo_utils import weights_init_normal
from trainer.trainer import Trainer
from dataset_utils.enums import Enums

def load_from_darknet53(config_path:str, weights_path:str, ):
    
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Select device for inference
    
    model = Darknet(config_path).to(device)
    model.apply(weights_init_normal)
    
    model.load_darknet_weights(weights_path)    

    return model                                                    

def load_model(config_path:str, weights_path:str=None):
    """Loads the yolo model from file.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :return: Returns model
    :rtype: Darknet
    """
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Select device for inference
    
    model = Darknet(config_path).to(device)
    model.apply(weights_init_normal)

    # If pretrained weights are specified, start from checkpoint or weight file
    if weights_path:
        if weights_path.endswith(".pth"):
            # Load checkpoint weights
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            # Load darknet weights
            model.load_darknet_weights(weights_path)
    return model
    
if __name__ == "__main__":    
    trainer_config = json.load(open('config/yolo_trainer.json'))
    
    darknet53_path = 'darknet53.conv.74'
    weights_pth_path = ""
    
    if os.path.exists(darknet53_path):
        model = load_from_darknet53('config/yolov3-KiTTi.cfg',darknet53_path)
    
    else:
        if os.path.exists(weights_pth_path):
            model = load_model('config/yolov3-KiTTi.cfg', weights_path=weights_pth_path)
        else:
            model = load_model('config/yolov3-KiTTi.cfg')
    
    ''' 
    TODO
    1. Add Logs after _init_dataloader, _init_optimizer 
    2. Implement Callbacks for model checkpointing 
    3. Complete Validation methods 
    '''

    ''' 
    Batch Size - Modified from yolo.cfg
    '''

    trainer = Trainer(
        model, 
        trainer_config['dataset_kwargs'],
        trainer_config['optimizer_kwargs'],
        trainer_config['trainer_kwargs'],
        trainer_config['lr_scheduler_kwargs']
    )
    
    trainer.train()