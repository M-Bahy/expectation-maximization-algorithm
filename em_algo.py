import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Load the original image
original_image_path = "/media/bahy/MEDO BAHY/CMS/Deep Learning/expectation-maximization-algorithm/Dataset/model.jpg"
image = Image.open(original_image_path)
image_array = np.array(image)

# Get dimensions
height, width, channels = image_array.shape

# Reshape to 2D array for GMM (each pixel as a 3D RGB vector)
pixels = image_array.reshape(-1, channels)

# Apply Gaussian Mixture Model with 2 components foreground and background
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(pixels)
labels = gmm.predict(pixels)

# Determine which cluster is the foreground (smaller cluster)
if np.sum(labels) < len(labels) / 2:
    foreground_label = 1
else:
    foreground_label = 0

# Create the binary segmentation mask
binary_mask = (labels == foreground_label).reshape(height, width)

# Generate the binary segmented image
binary_segmented_image = binary_mask * 255  # Convert mask to binary image (0 and 255)

# Create masked RGB image
masked_image = image_array.copy()
for c in range(channels):
    masked_image[:,:,c] = image_array[:,:,c] * binary_mask

# Display the original image and the segmented images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image_array)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Binary mask
axes[1].imshow(binary_segmented_image, cmap='gray')
axes[1].set_title('Binary Mask')
axes[1].axis('off')

# Masked RGB image
axes[2].imshow(masked_image)
axes[2].set_title('Segmented Image')
axes[2].axis('off')

plt.tight_layout()
plt.show()