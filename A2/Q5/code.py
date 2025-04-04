import open3d as o3d
import numpy as np
from scipy.stats import ortho_group
from copy import deepcopy
import os
import csv
import matplotlib.pyplot as plt

# Load 2 consecutive point clouds
pcd1 = o3d.io.read_point_cloud(r"selected_pcds\pointcloud_0000.pcd")
pcd2 = o3d.io.read_point_cloud(r"selected_pcds\pointcloud_0004.pcd")


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


# Run ICP
def run_icp(pcd1, pcd2, init_transform, threshold=0.1, method="point_to_point"):
    pcd1_down = deepcopy(pcd1).voxel_down_sample(0.05)
    pcd2_down = deepcopy(pcd2).voxel_down_sample(0.05)

    pcd1_down.estimate_normals()
    pcd2_down.estimate_normals()

    if method == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    init_eval = o3d.pipelines.registration.evaluate_registration(
        pcd1_down, pcd2_down, threshold, init_transform)

    reg_icp = o3d.pipelines.registration.registration_icp(
        pcd1_down, pcd2_down, threshold, init_transform, estimation)

    error = np.linalg.norm(reg_icp.transformation - init_transform)

    return {
        "fitness": reg_icp.fitness,
        "rmse": reg_icp.inlier_rmse,
        "error": error,
        "T": reg_icp.transformation
    }

# Get RANSAC Initial Guess
def get_ransac_initial_guess(pcd1, pcd2):
    voxel_size = 0.05
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    pcd2_down = pcd2.voxel_down_sample(voxel_size)

    pcd1_down.estimate_normals()
    pcd2_down.estimate_normals()

    pcd1_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd1_down, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=100))
    pcd2_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd2_down, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=100))

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd1_down, pcd2_down, pcd1_fpfh, pcd2_fpfh, True, 1.5 * voxel_size,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(1.5 * voxel_size)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result.transformation


# Initial Transforms
init_identity = np.eye(4)
init_random = np.eye(4)
init_random[:3, :3] = ortho_group.rvs(3)
init_random[:3, 3] = np.random.uniform(-0.1, 0.1, size=(3,))

results = []

# Identity Init, Point-to-Point
res1 = run_icp(pcd1, pcd2, init_identity, threshold=0.1, method="point_to_point")
results.append(("Identity", "point_to_point", 0.1, res1["fitness"], res1["rmse"], res1["error"]))

# Random Init, Point-to-Point
res2 = run_icp(pcd1, pcd2, init_random, threshold=0.1, method="point_to_point")
results.append(("Random", "point_to_point", 0.1, res2["fitness"], res2["rmse"], res2["error"]))

# Random Init, Point-to-Plane
res3 = run_icp(pcd1, pcd2, init_random, threshold=0.1, method="point_to_plane")
results.append(("Random", "point_to_plane", 0.1, res3["fitness"], res3["rmse"], res3["error"]))

# Identity Init, Point-to-Plane, lower threshold
res4 = run_icp(pcd1, pcd2, init_identity, threshold=0.05, method="point_to_plane")
results.append(("Identity", "point_to_plane", 0.05, res4["fitness"], res4["rmse"], res4["error"]))

# RANSAC Init, Point-to-Point
ransac_T = get_ransac_initial_guess(pcd1, pcd2)
res5 = run_icp(pcd1, pcd2, ransac_T, threshold=0.1, method="point_to_point")
results.append(("RANSAC", "point_to_point", 0.1, res5["fitness"], res5["rmse"], res5["error"]))



print(f"{'Init Guess':<12} | {'Method':<15} | {'Thresh':<7} | {'Fitness':<8} | {'RMSE':<8} | {'T Error':<8}")
print("-" * 70)
for name, method, threshold, fitness, rmse, error in results:
    print(f"{name:<12} | {method:<15} | {threshold:<7.3f} | {fitness:<8.4f} | {rmse:<8.4f} | {error:<8.4f}")


print("\nBest Transformation Matrix (from Identity Init + Point-to-Plane):\n", res4["T"])


# Downsample and estimate normals (needed for Point-to-Plane)
pcd1_down = deepcopy(pcd1).voxel_down_sample(0.05)
pcd2_down = deepcopy(pcd2).voxel_down_sample(0.05)

pcd1_down.estimate_normals()
pcd2_down.estimate_normals()

# Identity initialization
init_transform = np.eye(4)

# Run Point-to-Plane ICP with best settings
reg_icp = o3d.pipelines.registration.registration_icp(
    pcd1_down, pcd2_down, 0.05, init_transform,
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

# Get best estimated transformation
best_T = reg_icp.transformation
print("Best Transformation Matrix:\n", best_T)

# Apply transformation to the original (non-downsampled) point cloud
pcd1_transformed = deepcopy(pcd1).transform(best_T)

o3d.visualization.draw_geometries([
    pcd1_transformed.paint_uniform_color([1, 0, 0]),
    pcd2.paint_uniform_color([0, 1, 0])
])


print("Fitness:", reg_icp.fitness)
print("Inlier RMSE:", reg_icp.inlier_rmse)



base_path = r"selected_pcds"  # folder with the 100 .pcd files
file_names = [f"pointcloud_{i:04d}.pcd" for i in range(0, 400, 4)]  # pointcloud_0000.pcd to pointcloud_0396.pcd

global_pcds = []
global_poses = [np.eye(4)]
trajectory = [np.array([0, 0, 0])]

# ICP Parameters
voxel_size = 0.05
threshold = 0.1

# Run ICP
def run_icp(source, target, init=np.eye(4), threshold=0.1):
    return o3d.pipelines.registration.registration_icp(
        source, target, threshold, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

# Process All Consecutive Pairs 
for i in range(len(file_names) - 1):
    source_file = os.path.join(base_path, file_names[i])
    target_file = os.path.join(base_path, file_names[i+1])

    source = o3d.io.read_point_cloud(source_file).voxel_down_sample(voxel_size)
    target = o3d.io.read_point_cloud(target_file).voxel_down_sample(voxel_size)

    # Estimate normals (optional but helps)
    source.estimate_normals()
    target.estimate_normals()

    # Use identity as initial guess
    icp_result = run_icp(source, target, threshold=threshold)

    # Get last global transformation
    last_pose = global_poses[-1]
    current_pose = icp_result.transformation @ last_pose
    global_poses.append(current_pose)

    # Load original source cloud and transform it
    original = o3d.io.read_point_cloud(source_file)
    original.transform(last_pose)
    global_pcds.append(original)

    # Extract position from transformation
    translation = current_pose[:3, 3]
    trajectory.append(translation)

    print(f"Registered {file_names[i]} -> {file_names[i+1]} | Fitness: {icp_result.fitness:.4f}")

final_pcd = o3d.io.read_point_cloud(os.path.join(base_path, file_names[-1]))
final_pcd.transform(global_poses[-1])
global_pcds.append(final_pcd)

combined = global_pcds[0]
for pc in global_pcds[1:]:
    combined += pc

o3d.io.write_point_cloud("registered_map.pcd", combined)

with open("2022570_trajectory.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "z"])
    for point in trajectory:
        writer.writerow(point)

trajectory_np = np.array(trajectory)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_np[:, 0], trajectory_np[:, 1], trajectory_np[:, 2], marker='o')
ax.set_title("Estimated 3D Trajectory of TurtleBot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()