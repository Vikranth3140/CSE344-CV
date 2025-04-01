import cv2
import matplotlib.pyplot as plt
import os
from glob import glob




# 4.1


# Step 1: Load the first two images from the dataset
image_dir = 'panorama_dataset/'
image_paths = sorted(glob(os.path.join(image_dir, '*.png')))
img1_path, img2_path = image_paths[0], image_paths[1]

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Step 2: Initialize SIFT
sift = cv2.SIFT_create()

# Step 3: Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Step 4: Draw keypoints on images
img1_kp = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Convert for matplotlib (BGR → RGB)
img1_kp = cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB)
img2_kp = cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB)

# Step 5: Display
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.imshow(img1_kp)
plt.title('Keypoints in Image 1')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2_kp)
plt.title('Keypoints in Image 2')
plt.axis('off')

plt.tight_layout()
plt.show()

















# 4.2






import cv2
import numpy as np
import matplotlib.pyplot as plt

# Use gray images, keypoints and descriptors from earlier steps
# gray1, gray2, kp1, des1, kp2, des2 are assumed to already exist

# === BRUTEFORCE MATCHING ===
bf = cv2.BFMatcher()
bf_matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
bf_good = []
for m, n in bf_matches:
    if m.distance < 0.75 * n.distance:
        bf_good.append(m)

# Draw matches
bf_result = cv2.drawMatches(img1, kp1, img2, kp2, bf_good[:50], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# === FLANN MATCHING ===
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
flann_matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
flann_good = []
for m, n in flann_matches:
    if m.distance < 0.7 * n.distance:
        flann_good.append(m)

# Draw matches
flann_result = cv2.drawMatches(img1, kp1, img2, kp2, flann_good[:50], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# === DISPLAY RESULTS ===
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(bf_result, cv2.COLOR_BGR2RGB))
plt.title(f"BruteForce Matches: {len(bf_good)}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(flann_result, cv2.COLOR_BGR2RGB))
plt.title(f"FLANN Matches: {len(flann_good)}")
plt.axis('off')

plt.tight_layout()
plt.show()