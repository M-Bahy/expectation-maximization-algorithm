import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture



def GMM (original_image_path):

    image = Image.open(original_image_path)
    image_array = np.array(image)
    height, width, channels = image_array.shape
    # convert the image into pixels 2D array as expected by the model
    pixels = image_array.reshape(-1, channels)

    # 2 components foreground and background so the return is either 0 or 1
    # takes a 2D array were each row is a pixel and each column is a feature (color channel)
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(pixels)
    labels = gmm.predict(pixels)
    foreground_label = 1

    # create an image with the same size as the original
    # put True/1 in the pixels that are labeled as foreground , False/0 otherwise
    binary_mask = (labels == foreground_label).reshape(height, width)

    # multiply the mask by the original image to get the segmented image
    masked_image = image_array.copy()
    for c in range(channels):
        masked_image[:,:,c] = image_array[:,:,c] * binary_mask

    # covert the 0 and 1 to 0 and 255 by multiplying by 255
    binary_segmented_image = binary_mask * 255  

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

if __name__ == "__main__":
    original_image_path = "/media/bahy/MEDO BAHY/CMS/Deep Learning/expectation-maximization-algorithm/Dataset/model.jpg"
    GMM(original_image_path)