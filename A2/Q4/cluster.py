import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set directory
image_dir = "panorama_dataset"
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

# Load images
images = [cv2.imread(os.path.join(image_dir, f)) for f in image_files]

# Feature extraction via resizing + flattening
resized_images = [cv2.resize(img, (100, 100)).flatten() for img in images]

# PCA dimensionality reduction
pca = PCA(n_components=min(25, len(resized_images)))  # adjust to avoid error
features = pca.fit_transform(resized_images)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(features)

# Group images by cluster
clustered_images = {i: [] for i in range(3)}
for img, label in zip(images, labels):
    clustered_images[label].append(img)

# Stitching function
def stitch_images(img_list):
    stitcher = cv2.Stitcher_create()
    status, pano = stitcher.stitch(img_list)
    if status == cv2.Stitcher_OK:
        return pano
    else:
        print(f"⚠️ Stitching failed with status code {status}")
        return None

# Stitch and display
for cluster_id, imgs in clustered_images.items():
    if len(imgs) < 2:
        print(f"⚠️ Not enough images to stitch for cluster {cluster_id}")
        continue
    panorama = stitch_images(imgs)
    if panorama is not None:
        plt.figure(figsize=(14, 6))
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.title(f"Panorama - Cluster {cluster_id}")
        plt.axis("off")
        plt.show()