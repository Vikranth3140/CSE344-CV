import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


chessboard_size = (8, 6)
square_size = 30

# 3D object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

images = glob.glob('../chessboard_dataset/*.jpeg')

for file_name in images:
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Chessboard Corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners_refined)
        cv2.drawChessboardCorners(img, chessboard_size, corners_refined, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)
    else:
        print(f"Corners not found in {file_name}")

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# 3.1

# Intrinsic matrix (K)
print("Intrinsic Matrix (K):")
print(mtx)

# Extract intrinsic parameters
fx, fy = mtx[0, 0], mtx[1, 1]
print(f"Focal Lengths: fx = {fx:.2f}, fy = {fy:.2f} pixels")

cx, cy = mtx[0, 2], mtx[1, 2]
print(f"Principal Point: cx = {cx:.2f}, cy = {cy:.2f} pixels")

skew = mtx[0, 1]
print(f"Skew Parameter: {skew:.2f}")

print(f"Reprojection Error: {ret:.4f} pixels")


# 3.2

# Extract extrinsic parameters for the first 2 images
print("\nExtrinsic Parameters for First 2 Images:")
for i in range(min(2, len(rvecs))):
    R, _ = cv2.Rodrigues(rvecs[i])
    t = tvecs[i]
    print(f"\nImage {i+1}:")
    print("Rotation Matrix (R):")
    print(R)
    print("Translation Vector (t):")
    print(t)


# 3.3

# Extract distortion coefficients
print("\nDistortion Coefficients:")
print(dist)

os.makedirs('raw_images', exist_ok=True)
os.makedirs('undistorted_images', exist_ok=True)

# Undistort the first 5 images
for i, file_name in enumerate(images[:5]):
    img = cv2.imread(file_name)
    if img is None:
        print(f"Failed to load {file_name} for undistortion")
        continue

    raw_path = f'raw_images/raw_image_{i+1:02d}.jpeg'
    cv2.imwrite(raw_path, img)

    # Undistort the image
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

    undistorted_path = f'undistorted_images/undistorted_image_{i+1:02d}.jpeg'
    cv2.imwrite(undistorted_path, undistorted_img)


# 3.4

reprojection_errors = []

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    reprojection_errors.append(error)

# individual errors
for i, err in enumerate(reprojection_errors):
    print(f"Image {i+1:02d} - Reprojection Error: {err:.4f} pixels")

# mean and standard deviation
mean_error = np.mean(reprojection_errors)
std_error = np.std(reprojection_errors)
print(f"\nMean Reprojection Error: {mean_error:.4f} pixels")
print(f"Standard Deviation: {std_error:.4f} pixels")

plt.figure(figsize=(10, 5))
plt.bar(range(1, len(reprojection_errors)+1), reprojection_errors, color='royalblue')
plt.axhline(mean_error, color='red', linestyle='--', label=f"Mean = {mean_error:.4f}")
plt.title("Per-Image Reprojection Error")
plt.xlabel("Image Index")
plt.ylabel("Reprojection Error (pixels)")
plt.xticks(range(1, len(reprojection_errors)+1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reprojection_error_plot.png")
plt.show()

# 3.5


os.makedirs('reprojection_visuals', exist_ok=True)

for i, file_name in enumerate(images):
    img = cv2.imread(file_name)
    if img is None:
        print(f"Failed to load {file_name}")
        continue

    # Project object points using estimated parameters
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

    # Draw original detected corners (green) and reprojected corners (red)
    for p1, p2 in zip(imgpoints[i], imgpoints2):
        pt1 = tuple(np.round(p1.ravel()).astype(int))  # Detected
        pt2 = tuple(np.round(p2.ravel()).astype(int))  # Reprojected
        cv2.circle(img, pt1, 4, (0, 255, 0), -1)  # Green
        cv2.circle(img, pt2, 2, (0, 0, 255), -1)  # Red

    # Save image
    out_path = f"reprojection_visuals/reprojection_comparison_{i+1:02d}.jpeg"
    cv2.imwrite(out_path, img)


# 3.6

plane_normals = []

for i in range(len(rvecs)):
    R, _ = cv2.Rodrigues(rvecs[i])         # Convert to rotation matrix
    normal_ci = R[:, 2]                    # Third column is the Z-axis of the checkerboard in camera frame
    plane_normals.append(normal_ci)
    print(f"Image {i+1:02d} - Plane Normal (Camera Frame): {normal_ci}")