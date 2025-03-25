import cv2
import numpy as np
import glob
import json

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
images = glob.glob('chessboard_dataset/*.jpeg')
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
    "distortion": []
}

with open('calibration_provided.json', 'w') as f:
    json.dump(calibration_data, f, indent=4)

print("Saved extrinsic parameters to calibration_provided.json")