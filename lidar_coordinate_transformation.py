import numpy as np
import cv2

def upscale_with_bilateral_filter(depth_map, reflectance_map):
    # Apply bilateral filtering
    depth_map_filtered = cv2.bilateralFilter(depth_map, d=9, sigmaColor=75, sigmaSpace=75)
    reflectance_map_filtered = cv2.bilateralFilter(reflectance_map, d=9, sigmaColor=75, sigmaSpace=75)

    return depth_map_filtered, reflectance_map_filtered

def read_image(image_path:str):
    return cv2.imread(image_path)
    
def read_velodyne_bin(file_path):
    # Load the .bin file
    point_cloud = np.fromfile(file_path, dtype=np.float32)
        
    # Reshape the array to (N, 4), where N is the number of points
    point_cloud = point_cloud.reshape(-1, 4)    
    # Extract x, y, z coordinates
    return point_cloud

def read_calibration_file(calib_file_path):
    calibration_dict = {}
    
    with open(calib_file_path, 'r') as f:
        for line in f.readlines():
            if line != '\n':
                key, value = line.split(':')
                calibration_dict[key.strip()] = np.fromstring(
                    value, sep=' '
                ).reshape(-1, 3)
                
    calibration_dict['Tr_velo_to_cam'] = np.hstack(
        (calibration_dict['Tr_velo_to_cam'], np.array([[0], [0], [0], [1]]))
    ).T
        
    return calibration_dict

def transform_lidar_to_camera(lidar_points:np.array, tr_velo_to_cam:np.array):
    
    if lidar_points.shape[-1] > 3:
        lidar_points = lidar_points[:, :3]
    
    lidar_points_homogeneous = np.hstack(
        (lidar_points, np.ones((lidar_points.shape[0], 1)))
    )
    
    lidar_transformed_cam = lidar_points_homogeneous @ tr_velo_to_cam
    lidar_transformed_cam = lidar_transformed_cam[:,:3]
    
    return lidar_transformed_cam

def project_to_image_coordinates(lidar_transformed_cam:np.array, projection_matrix:np.array):
    
    lidar_transformed_cam_homo = np.hstack(
        (lidar_transformed_cam, np.ones((lidar_transformed_cam.shape[0], 1)))
    )
    
    image_coordinates_homo = projection_matrix.T @ lidar_transformed_cam_homo.T
        
    x_img = image_coordinates_homo[0]/image_coordinates_homo[2]
    y_img = image_coordinates_homo[1]/image_coordinates_homo[2]
        
    image_coordinates = np.vstack((x_img, y_img)).T
    
    image_coordinates = np.concatenate((
        [image_coordinates, np.expand_dims(lidar_transformed_cam[:, -1], axis=1)]
    ), axis=1)
            
    return image_coordinates

def create_maps(image_coordinates:np.array, image_arr:np.array):
    
    depth_map = np.zeros_like(image_arr, dtype=np.float32)
    reflectance_map = np.zeros_like(image_arr, dtype=np.float32)
    
    image_shape = image_arr.shape
    
    def fill_maps(x_img, y_img, depth_c, intensity_l):
        x_pixel = int(round(x_img))
        y_pixel = int(round(y_img))
        
        # Check if the projected pixel coordinates are within image bounds
        if 0 <= x_pixel < image_shape[1] and 0 <= y_pixel < image_shape[0]:
            depth_map[y_pixel, x_pixel] += depth_c
            reflectance_map[y_pixel, x_pixel] += intensity_l
            
    for point in image_coordinates:
        x_img, y_img, z_c, I_l = point
        fill_maps(x_img, y_img, z_c, I_l)
        
    return upscale_with_bilateral_filter(depth_map, reflectance_map)    

def save_as_png(depth_map, reflectance_map, depth_file='depth_map.png', reflectance_file='reflectance_map.png'):
    # Normalize the depth map to 0-255
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)  # Convert to unsigned 8-bit integer

    # Normalize the reflectance map to 0-255
    reflectance_normalized = cv2.normalize(reflectance_map, None, 0, 255, cv2.NORM_MINMAX)
    reflectance_normalized = reflectance_normalized.astype(np.uint8)  # Convert to unsigned 8-bit integer

    # Save the images
    cv2.imwrite(depth_file, depth_normalized)
    cv2.imwrite(reflectance_file, reflectance_normalized)
            
def transform_lidar_sample(lidar_bin_file_path:str, calib_file_path:str, image_fn:str):
    
    point_cloud_array = read_velodyne_bin(lidar_bin_file_path) #(X_l, Y_l, Z_l, I_l)
    calibration_dict = read_calibration_file(calib_file_path)
    image_array = read_image(image_fn)
        
    lidar_transformed_cam = transform_lidar_to_camera(
        point_cloud_array, calibration_dict['Tr_velo_to_cam']
    ) #(X_c, Y_c, Z_c)

    image_coordinates = project_to_image_coordinates(
        lidar_transformed_cam, calibration_dict['P2']
    ) #(X_img, Y_img, Z_c)
                    
    #(X_img, Y_img, Z_c, I_l)
    image_coordinates = np.concatenate([image_coordinates, 
                                               np.expand_dims(point_cloud_array[:,-1], axis=1)], axis=1) 
    
    depth_map, reflectance_map = create_maps(image_coordinates, image_array)
        
    save_as_png(depth_map, reflectance_map)

if __name__ == "__main__":
    # Path to your .bin file
    lidar_bin_file_path = "dev_datakit/velodyne/training/000044.bin"  #CHANGE
    calib_file_path = "calibration/training/calib/000044.txt" #CHANGE
    image_path = "dev_datakit/image_left/training/000044.png" #CHANGE
    
    transform_lidar_sample(
        lidar_bin_file_path, calib_file_path, image_path
    )
    
    