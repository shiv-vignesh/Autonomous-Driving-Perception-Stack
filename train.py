import json
from model.yolo import Darknet
from trainer.trainer import Trainer

def create_model(yolo_cfg:str):
    return Darknet(
        yolo_cfg
    )
    
if __name__ == "__main__":    
    trainer_config = json.load(open('config/yolo_trainer.json'))
    model = create_model('config/yolov3-KiTTi.cfg')
    
    trainer = Trainer(
        model, 
        trainer_config['dataset_kwargs'],
        trainer_config['optimizer_kwargs'],
        trainer_config['trainer_kwargs'],
        trainer_config['lr_scheduler_kwargs']
    )
    
    trainer.train()