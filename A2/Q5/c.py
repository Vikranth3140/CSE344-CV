import open3d as o3d
import numpy as np
from scipy.stats import ortho_group

# Load 2 consecutive point clouds
pcd1 = o3d.io.read_point_cloud("selected_pcds\pointcloud_0000.pcd")
pcd2 = o3d.io.read_point_cloud("selected_pcds\pointcloud_0004.pcd")

# Downsample for faster processing
pcd1_down = pcd1.voxel_down_sample(voxel_size=0.05)
pcd2_down = pcd2.voxel_down_sample(voxel_size=0.05)

# Estimate normals (important for ICP)
pcd1_down.estimate_normals()
pcd2_down.estimate_normals()

# Generate a random orthonormal rotation + small translation
R = ortho_group.rvs(dim=3)
t = np.array([[0.2], [0.1], [0.0]])  # small random translation
init_transform = np.eye(4)
init_transform[:3, :3] = R
init_transform[:3, 3:] = t

# Evaluate initial alignment
init_eval = o3d.pipelines.registration.evaluate_registration(
    pcd1_down, pcd2_down, 0.1, init_transform
)
print("Initial Alignment")
print("Fitness:", init_eval.fitness)
print("Inlier RMSE:", init_eval.inlier_rmse)

# Run ICP
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1_down, pcd2_down, 0.1, init_transform,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print("\nAfter ICP")
print("Estimated Transformation:\n", reg_p2p.transformation)
print("Fitness:", reg_p2p.fitness)
print("Inlier RMSE:", reg_p2p.inlier_rmse)

# Visualize alignment
pcd1_down.transform(reg_p2p.transformation)
o3d.visualization.draw_geometries([pcd1_down.paint_uniform_color([1, 0, 0]), 
                                   pcd2_down.paint_uniform_color([0, 1, 0])])