# Multi-Modal 2D Object Detection - Spatial-IL Fusion Pipeline

This project implements a multi-modal 2D object detection system that fuses LiDAR point cloud data with camera images using the KITTI dataset.

## Highlights
- Pointnet Backbone - LiDAR point cloud processing. 
- YOLOv3 Backbone - Image input processing. 
- Adaptive Fusion Backbone that combines Lidar and Image features. 
    - Project LiDAR point cloud features from `(1024, N)` to `[(yolo_grid_x, yolo_grid_y, 1024)]`. 
    - Process by a Adaptive Fusion block for each grid features. 
- Produce Detections. 

## Acknowledgments

This project was developed as part of the CSCI-739 term project offered by the Rochester Institute of Technology (RIT). The project was completed under the guidance of <b>[Dr. Fawad Ahmad](https://fawadahm.github.io/)</b> and in collaboration with <b>[Ashutosh Kumar](https://github.com/ashu1069)</b> and <b>[Sarthak Kaushal](https://github.com/Sarthakaushal)</b>.

### Code Contributions
- Part of the codebase for YOLOv3 implementation is adapted from [PyTorch-YOLOv3 by Erik Lindernoren](https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/master).
  - Please also refer to the following paper:
    ```
    @article{redmon2018yolov3,
      title={YOLOv3: An Incremental Improvement},
      author={Redmon, Joseph and Farhadi, Ali},
      journal={arXiv preprint arXiv:1804.02767},
      year={2018}
    }
    ```

### Dataset
- This project utilized the [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/) for training and evaluation purposes. 
  ```
  Geiger, A., Lenz, P., Stiller, C., & Urtasun, R. (2013). 
  Vision meets robotics: The kitti dataset. 
  The International Journal of Robotics Research, 32(11), 1231-1237.
  ```


### Data Preprocessing
![data_preprocessing_pipeline](samples/datapreproessing_steps_visualize.png)

### Multi-Modal Pipeline - Spatial-IL Fusion Pipeline
![Spatial-IL Fusion Pipeline](samples/Proposed-Methodology-Complete.png)

### Detection Outputs
![Spatial-IL Fusion Pipeline Detection](samples/000152_detection.png)
![Spatial-IL Fusion Pipeline Detection](samples/000181_detection.png)
![Spatial-IL Fusion Pipeline Detection](samples/000618_detection.png)

## Usage 

1. Prepare the KITTI dataset:

    - Download the KITTI object detection dataset
    - Organize the data in the following structure:

            data/
                ├── calibration/
                │   └── training/
                │       └── calib/
                ├── left_images/
                │   └── training/
                │       └── image_2/
                ├── labels/
                │   └── training/
                │       └── label_2/
                └── velodyne/
                    └── training/
                        └── velodyne/

## Configuration File: `config/yolo_pointnet_fusion_trainer.json`

Customize the pipeline by editing the following sections in the JSON file:

### 1. **Dataset Configuration (`dataset_kwargs`)**

#### Training Dataset
Specify paths and parameters for training & Validation data:
```json
"trainer_dataset_kwargs": {
    "lidar_dir": "data/velodyne/training/velodyne",
    "calibration_dir": "data/calibration/training/calib",
    "left_image_dir": "data/left_images/training/image_2",
    "labels_dir": "data/labels/training/label_2",
    "shuffle": true,
    "apply_augmentation": false
}

"validation_dataset_kwargs": {
    "lidar_dir": "data/velodyne/validation/velodyne",
    "calibration_dir": "data/calibration/validation/calib",
    "left_image_dir": "data/left_images/validation/image_2",
    "labels_dir": "data/labels/validation/label_2",
    "shuffle": false,
    "apply_augmentation": false
}
``` 
#### Define the PointNet backbone settings:
```json
"pointnet_kwargs": {
    "num_points": 75000,
    "num_global_feats": 1024
}
```

#### Adjust fusion-related parameters:
```json
"adaptive_fusion_kwargs": {
    "fusion_type": "residual",
    "transform_image_features": false
}
```

# YOLO PointNet Fusion Training Pipeline

This project implements a training pipeline for fusing YOLO-based object detection with PointNet for point cloud processing. Below is an explanation of the `trainer_kwargs` section in the configuration file.

---

## Trainer Configuration (`trainer_kwargs`)

The `trainer_kwargs` section in the JSON configuration file controls the behavior of the training process. Below is an explanation of each key parameter.

### Key Parameters

1. **`output_dir`**
   - Specifies the directory where training outputs (e.g., model checkpoints, logs) will be saved.
   - Example:  
     ```json
     "output_dir": "Robust-Spatial-Fusion-Pipeline-3"
     ```

2. **`is_training`**
   - Indicates whether the pipeline is in training mode.  
   - Set to `true` for training and `false` for evaluation or testing.  
   - Example:  
     ```json
     "is_training": true
     ```

3. **`first_val_epoch`**
   - The epoch at which validation should start.  
   - Example:  
     ```json
     "first_val_epoch": 0
     ```

4. **`metric_eval_mode`**
   - Determines how metrics are calculated. Common modes:  
     - `strict`: Enforces stricter criteria for evaluation.
   - Example:  
     ```json
     "metric_eval_mode": "strict"
     ```

5. **`metric_average_mode`**
   - Specifies the mode for averaging metrics across classes.  
     - `macro`: Averages metrics equally across all classes.
   - Example:  
     ```json
     "metric_average_mode": "macro"
     ```

6. **`epochs`**
   - The total number of training epochs.  
   - Example:  
     ```json
     "epochs": 60
     ```

7. **`monitor_train`**
   - Toggles monitoring of training metrics.  
   - Set to `true` to log training performance.  
   - Example:  
     ```json
     "monitor_train": true
     ```

8. **`monitor_val`**
   - Toggles monitoring of validation metrics.  
   - Set to `true` to log validation performance.  
   - Example:  
     ```json
     "monitor_val": true
     ```

9. **`gradient_clipping`**
   - Limits the magnitude of gradients to stabilize training.  
   - Example:  
     ```json
     "gradient_clipping": 1.0
     ```

10. **`yolo_device_id`**
    - Specifies the device for running the YOLO model (e.g., `cuda:0` for the first GPU).  
    - Example:  
      ```json
      "yolo_device_id": "cuda:0"
      ```

11. **`pointnet_device_id`**
    - Specifies the device for running the PointNet model.  
    - Example:  
      ```json
      "pointnet_device_id": "cuda:1"
      ```

12. **`checkpoint_idx`**
    - Indicates the index of the checkpoint to load for resuming training.  
    - Example:  
      ```json
      "checkpoint_idx": 10
      ```

13. **`gradient_accumulation_steps`**
    - The number of steps to accumulate gradients before performing a weight update. Useful for training with limited GPU memory.  
    - Example:  
      ```json
      "gradient_accumulation_steps": 4
      ```

14. **`compute_feature_alignment`**
    - Enables feature alignment computation between YOLO and PointNet.  
    - Example:  
      ```json
      "compute_feature_alignment": false
      ```

15. **`yolo_lr_burn_in`**
    - Toggles a learning rate warm-up phase for YOLO training.  
    - Example:  
      ```json
      "yolo_lr_burn_in": true
      ```

16. **`robustness_augmentations`**
    - Specifies a list of data augmentations to improve model robustness.  
    - Example:  
      ```json
      "robustness_augmentations": ["SaltPapperNoise", "pixelate"]
      ```

---

### Example `trainer_kwargs` Configuration

Here is a complete example of a `trainer_kwargs` configuration:

```json
"trainer_kwargs": {
    "output_dir": "Robust-Spatial-Fusion-Pipeline-3",
    "is_training": true,
    "first_val_epoch": 0,
    "metric_eval_mode": "strict",
    "metric_average_mode": "macro",
    "epochs": 60,
    "monitor_train": true,
    "monitor_val": true,
    "gradient_clipping": 1.0,
    "yolo_device_id": "cuda:0",
    "pointnet_device_id": "cuda:1",
    "checkpoint_idx": 10,
    "gradient_accumulation_steps": 4,
    "compute_feature_alignment": false,
    "yolo_lr_burn_in": true,
    "robustness_augmentations": ["SaltPapperNoise", "pixelate"]
}


