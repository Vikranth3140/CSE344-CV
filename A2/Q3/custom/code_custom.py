import cv2
import numpy as np
import glob
import json
import os

# Chessboard size (number of inner corners)
CHESSBOARD_SIZE = (8, 6)  # (width, height) in corners
SQUARE_SIZE = 30  # Size of each square in mm (assumed)

# Prepare 3D object points (world coordinates of corners)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points
objpoints = []
imgpoints = []

# Load images from the provided dataset
images = glob.glob('../custom_dataset/*.jpg')
if len(images) != 25:
    print(f"Warning: Found {len(images)} images, expected 25")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                          criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners_refined)
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners_refined, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)
    else:
        print(f"Corners not found in {fname}")

cv2.destroyAllWindows()
print(f"Processed {len(imgpoints)} images successfully")

# Calibrate the camera
if len(objpoints) == 0:
    print("No images with detected corners. Cannot calibrate.")
    exit()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

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

# Save intrinsic parameters
with open('intrinsic_params_provided.txt', 'w') as f:
    f.write("Intrinsic Parameters (Provided Dataset):\n")
    f.write(f"Intrinsic Matrix (K):\n{mtx}\n\n")
    f.write(f"Focal Lengths: fx = {fx:.2f}, fy = {fy:.2f} pixels\n")
    f.write(f"Principal Point: cx = {cx:.2f}, cy = {cy:.2f} pixels\n")
    f.write(f"Skew Parameter: {skew:.2f}\n")
    f.write(f"Reprojection Error: {ret:.4f} pixels\n")

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

# Save extrinsics
with open('extrinsic_params_provided.txt', 'w') as f:
    f.write("Extrinsic Parameters (Provided Dataset):\n")
    for i in range(min(2, len(rvecs))):
        R, _ = cv2.Rodrigues(rvecs[i])
        t = tvecs[i]
        f.write(f"\nImage {i+1}:\n")
        f.write("Rotation Matrix (R):\n")
        f.write(f"{R}\n")
        f.write("Translation Vector (t):\n")
        f.write(f"{t}\n")

# Extract distortion coefficients
print("\nDistortion Coefficients:")
print(dist)

# Save distortion coefficients
with open('distortion_params_provided.txt', 'w') as f:
    f.write("Distortion Coefficients (Provided Dataset):\n")
    f.write(f"{dist}\n")

# Create directories for raw and undistorted images
os.makedirs('raw_images', exist_ok=True)
os.makedirs('undistorted_images', exist_ok=True)

# Undistort the first 5 images
for i, fname in enumerate(images[:5]):
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load {fname} for undistortion")
        continue

    # Save raw image
    raw_path = f'raw_images/raw_image_{i+1:02d}.jpeg'
    cv2.imwrite(raw_path, img)

    # Undistort the image
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

    # Save undistorted image
    undistorted_path = f'undistorted_images/undistorted_image_{i+1:02d}.jpeg'
    cv2.imwrite(undistorted_path, undistorted_img)

    print(f"Saved raw and undistorted versions of {fname}")

# Prepare JSON
intrinsics = {
    "fx": float(mtx[0, 0]),
    "fy": float(mtx[1, 1]),
    "cx": float(mtx[0, 2]),
    "cy": float(mtx[1, 2]),
    "skew": float(mtx[0, 1])
}

extrinsics = []
for i in range(min(2, len(rvecs))):
    R, _ = cv2.Rodrigues(rvecs[i])
    t = tvecs[i]
    extrinsics.append({
        "image": f"image_{i+1:02d}.jpeg",
        "rotation_matrix": R.tolist(),
        "translation_vector": t.flatten().tolist()
    })

calibration_data = {
    "intrinsics": intrinsics,
    "extrinsics": extrinsics,
    "distortion": dist.flatten().tolist()
}

with open('calibration_provided.json', 'w') as f:
    json.dump(calibration_data, f, indent=4)

print("Saved distortion coefficients to calibration_provided.json")



















# Reprojection - 3.4



import matplotlib.pyplot as plt

# Compute reprojection error for each image
reprojection_errors = []

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    reprojection_errors.append(error)

# Report individual errors
for i, err in enumerate(reprojection_errors):
    print(f"Image {i+1:02d} - Reprojection Error: {err:.4f} pixels")

# Compute mean and standard deviation
mean_error = np.mean(reprojection_errors)
std_error = np.std(reprojection_errors)
print(f"\nMean Reprojection Error: {mean_error:.4f} pixels")
print(f"Standard Deviation: {std_error:.4f} pixels")

# Plot bar chart of errors
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





# corners detected - 3.5


# Directory to save comparison images
os.makedirs('reprojection_visuals', exist_ok=True)

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load {fname}")
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
    print(f"Saved reprojection overlay for Image {i+1}")








# checkerboard plane normals - 3.6



# Compute and save checkerboard plane normals in camera frame
plane_normals = []

for i in range(len(rvecs)):
    R, _ = cv2.Rodrigues(rvecs[i])         # Convert to rotation matrix
    normal_ci = R[:, 2]                    # Third column is the Z-axis of the checkerboard in camera frame
    plane_normals.append(normal_ci)
    print(f"Image {i+1:02d} - Plane Normal (Camera Frame): {normal_ci}")

# Optional: save to file
with open('checkerboard_normals_camera_frame.txt', 'w') as f:
    f.write("Checkerboard Plane Normals (Camera Frame):\n")
    for i, n in enumerate(plane_normals):
        f.write(f"Image {i+1:02d}: {n.tolist()}\n")