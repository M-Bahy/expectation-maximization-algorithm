import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Load the original image and convert it to grayscale
original_image_path = "/media/bahy/MEDO BAHY/CMS/Deep Learning/expectation-maximization-algorithm/Dataset/model.jpg"  # Update with your image path
image = Image.open(original_image_path).convert('L')  # Convert to grayscale
image_array = np.array(image)
height, width = image_array.shape
pixels = image_array.reshape(-1, 1)  # Reshape to 2D array for GMM

# Apply Gaussian Mixture Model with 2 components
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

# Display the original grayscale image and the binary segmented image
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_array, cmap='gray')
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

axes[1].imshow(binary_segmented_image, cmap='gray')
axes[1].set_title('Binary Segmented Image')
axes[1].axis('off')

plt.show()