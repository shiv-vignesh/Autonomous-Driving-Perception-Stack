import numpy as np
import open3d as o3d

def read_velodyne_bin(file_path):
    # Load the .bin file
    point_cloud = np.fromfile(file_path, dtype=np.float32)
        
    # Reshape the array to (N, 4), where N is the number of points
    point_cloud = point_cloud.reshape(-1, 4)    
    # Extract x, y, z coordinates
    xyz = point_cloud[:, :3]
    return xyz

def visualize_point_cloud(xyz):
    output_file_path = "output_point_cloud.ply"
    
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # Set points
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # # Visualize
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(output_file_path, pcd)

if __name__ == "__main__":
    # Path to your .bin file
    bin_file_path = "dev_datakit/velodyne/training/000044.bin"
    
    # Read and visualize
    point_cloud_xyz = read_velodyne_bin(bin_file_path)
    visualize_point_cloud(point_cloud_xyz)
