import torch, time
from torch.utils.data import DataLoader

from dataset_utils.kitti_2d_objectDetect import Kitti2DObjectDetectDataset, KittiLidarFusionCollateFn
from trainer.loss import compute_loss
from model.yolo import Darknet

if __name__ == "__main__":
    
    yolo = Darknet(
        'config/yolov3-KiTTi.cfg'
    )
    
    dataset = Kitti2DObjectDetectDataset(
        lidar_dir='data/dev_datakit/velodyne/training',
        calibration_dir='data/calibration/training/calib',
        left_image_dir='data/dev_datakit/image_left/training',
        labels_dir='data/left_image_labels/training/label_2'
    )
        
    dataloader = DataLoader(
        dataset, 
        batch_size=8,
        collate_fn=KittiLidarFusionCollateFn(
            image_resize=(416, 416)
        ),
        shuffle=True
    )
    
    yolo.to('cuda')
    
    for data_items in dataloader:
        for k,v in data_items.items():
            if torch.is_tensor(v):                    
                data_items[k] = v.to('cuda')        
        outputs = yolo(data_items['images'])
        loss = compute_loss(outputs, data_items['targets'], yolo)
        print(loss)
        # time.sleep(100)
        # print(data_items['targets'].size())
        # exit(1)
        
        # print(data_items['bboxes'][0].__len__())
        # print(data_items['class_labels'][0].__len__())
        exit(1)