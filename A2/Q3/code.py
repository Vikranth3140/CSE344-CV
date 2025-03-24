import cv2
import numpy as np
import glob

# Chessboard size (number of inner corners)
CHESSBOARD_SIZE = (8, 6)  # (width, height) in corners, based on 8x6 squares
SQUARE_SIZE = 30  # Size of each square in mm (assumed; adjust if specified)

# Prepare 3D object points (world coordinates of corners)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # Scale by square size

# Arrays to store object points and image points
objpoints = []  # 3D points in world coordinates
imgpoints = []  # 2D points in image plane

# Load images from the provided dataset
images = glob.glob('chessboard_dataset/*.jpeg')  # Adjust path to your folder
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

    if ret:  # Corners found
        objpoints.append(objp)
        # Refine corner positions
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                          criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners_refined)
        # Visualize corners (optional, for debugging)
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners_refined, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)  # Display for 500ms
    else:
        print(f"Corners not found in {fname}")

cv2.destroyAllWindows()
print(f"Processed {len(imgpoints)} images successfully")