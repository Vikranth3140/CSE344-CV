import cv2
import matplotlib.pyplot as plt
import os
from glob import glob

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