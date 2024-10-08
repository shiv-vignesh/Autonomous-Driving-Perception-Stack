import numpy as np 
import cv2
import matplotlib.pyplot as plt

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
                )
                
    return calibration_dict

def transform_lidar_to_camera(point_cloud_array:np.array, calibration_dict:dict):
    
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

def create_maps(projected_points:np.array, y_max, x_max, intensities:np.array, depths:np.array):

    reflectance_map = np.zeros((y_max, x_max), dtype=np.float32)
    depth_map = np.zeros((y_max, x_max), dtype=np.float32)    
    
    # Fill in the reflectance and depth maps
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
    
    return upscale_with_bilateral_filter(depth_map, reflectance_map)

def upscale_with_bilateral_filter(depth_map, reflectance_map):
    # Apply bilateral filtering
    depth_map_filtered = cv2.bilateralFilter(depth_map, d=9, sigmaColor=75, sigmaSpace=75)
    reflectance_map_filtered = cv2.bilateralFilter(reflectance_map, d=9, sigmaColor=75, sigmaSpace=75)

    return depth_map_filtered, reflectance_map_filtered

def visualize_maps(reflectance_map: np.array, depth_map: np.array):
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.title('Reflectance Map')
    plt.imshow(reflectance_map, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Depth Map')
    plt.imshow(depth_map, cmap='jet')
    plt.axis('off')

    # plt.show()
    plt.savefig('depth_reflectance.png', bbox_inches='tight', pad_inches=0)

def save_as_png(depth_map, reflectance_map, depth_file='depth_map.png', reflectance_file='reflectance_map.png'):
    # Normalize the depth map to 0-255
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)  # Convert to unsigned 8-bit integer

    # Normalize the reflectance map to 0-255
    reflectance_normalized = cv2.normalize(reflectance_map, None, 0, 255, cv2.NORM_MINMAX)
    reflectance_normalized = reflectance_normalized.astype(np.uint8)  # Convert to unsigned 8-bit integer

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    reflectance_colored = cv2.applyColorMap(reflectance_normalized, cv2.COLORMAP_JET)

    # Save the images
    cv2.imwrite(depth_file, depth_colored)
    cv2.imwrite(reflectance_file, reflectance_colored)

def transform_lidar_sample(lidar_bin_file_path:str, calib_file_path:str, image_fn:str):
    
    point_cloud_array = read_velodyne_bin(lidar_bin_file_path) #(X_l, Y_l, Z_l, I_l)
    calibration_dict = read_calibration_file(calib_file_path)
    image_array = read_image(image_fn)
    
    points_2d, depths = transform_lidar_to_camera(
        point_cloud_array, calibration_dict
    )
    
    x_min, y_min = 0, 0
    x_max, y_max = image_array.shape[1], image_array.shape[0]
    clip_distance = 2.0
    
    fov_inds = (
            (points_2d[:, 0] < x_max)
            & (points_2d[:, 0] >= x_min)
            & (points_2d[:, 1] < y_max)
            & (points_2d[:, 1] >= y_min)
    )    
    
    fov_inds = fov_inds & (
                point_cloud_array[:, 0] > clip_distance)
    
    imgfov_pc_velo = point_cloud_array[fov_inds, :]
    projected_points = points_2d[fov_inds]

    reflectance_map = np.zeros((y_max, x_max), dtype=np.float32)
    depth_map = np.zeros((y_max, x_max), dtype=np.float32)    
    
    depth_map, reflectance_map = create_maps(
        projected_points, y_max, x_max,
        point_cloud_array[fov_inds, 3],
        point_cloud_array[fov_inds, 2]
        # depths[fov_inds]
    )
    
    # visualize_maps(reflectance_map, depth_map)
    save_as_png(depth_map, reflectance_map)
                
if __name__ == "__main__":
    # Path to your .bin file
    lidar_bin_file_path = "dev_datakit/velodyne/training/000044.bin"  #CHANGE
    calib_file_path = "calibration/training/calib/000044.txt" #CHANGE
    image_path = "dev_datakit/image_left/training/000044.png" #CHANGE
    
    transform_lidar_sample(
        lidar_bin_file_path, calib_file_path, image_path
    )    