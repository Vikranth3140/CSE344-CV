import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from glob import glob

# Step 1: Load all images from panorama_dataset/
image_dir = 'panorama_dataset/'
image_paths = glob(os.path.join(image_dir, '*.png'))  # or .jpg

images = []
histograms = []
filenames = []

for path in image_paths:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)
    filenames.append(os.path.basename(path))

    # Compute RGB color histogram (8x8x8 bins)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    histograms.append(hist)

# Step 2: Perform KMeans clustering (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(histograms)

# Step 3: Organize and display images by cluster
clusters = {i: [] for i in range(3)}
for idx, label in enumerate(labels):
    clusters[label].append((filenames[idx], images[idx]))

# Step 4: Display results
for cluster_id in clusters:
    print(f"\nCluster {cluster_id}: {[name for name, _ in clusters[cluster_id]]}")
    sample_imgs = clusters[cluster_id][:min(4, len(clusters[cluster_id]))]  # show first 4 images per cluster
    fig, axs = plt.subplots(1, len(sample_imgs), figsize=(15, 5))
    if len(sample_imgs) == 1:
        axs = [axs]
    for ax, (name, img) in zip(axs, sample_imgs):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis('off')
    plt.tight_layout()
    plt.show()