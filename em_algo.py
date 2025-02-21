import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import cv2
import os

def segment_image(image_path, expected_path=None, n_components=2, n_init=5, save_plot=True):
    # Check if image path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read the image with error handling
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    try:
        # Reshape the image for GMM
        rows, cols, channels = image.shape
        reshaped_image = image.reshape(-1, channels)
        
        # Initialize and fit GMM with explicit parameters
        gmm = GaussianMixture(
            n_components=n_components,
            n_init=n_init,
            random_state=42,
            covariance_type='full',
            max_iter=100,
            # n_jobs=1  # Use single thread to avoid threadpool issues
        )
        
        # Fit and predict
        gmm.fit(reshaped_image)
        labels = gmm.predict(reshaped_image)
        segmented = labels.reshape(rows, cols)
        
        # Create and apply mask
        mask = np.zeros_like(segmented)
        mask[segmented == 1] = 1
        
        # Create segmented image
        segmented_image = image.copy()
        for c in range(channels):
            segmented_image[:,:,c] = image[:,:,c] * mask
            
        # Plotting
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        if expected_path and os.path.exists(expected_path):
            expected = cv2.imread(expected_path)
            expected = cv2.cvtColor(expected, cv2.COLOR_BGR2RGB)
            plt.subplot(132)
            plt.imshow(expected)
            plt.title('Expected Output')
            plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(segmented_image , cmap='gray')
            plt.title('Segmented Image')
            plt.axis('off')
        else:
            plt.subplot(132)
            plt.imshow(segmented_image)
            plt.title('Segmented Image')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_plot:
            output_dir = os.path.dirname(image_path)
            plt.savefig(os.path.join(output_dir, 'segmentation_result.png'))
        else:
            plt.show()
        
        plt.close()  # Clean up
        
        return segmented_image
    
    except Exception as e:
        print(f"Error during image segmentation: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        image_path = "/media/bahy/MEDO BAHY/CMS/Deep Learning/expectation-maximization-algorithm/Dataset/model.jpg"
        expected_path = "/media/bahy/MEDO BAHY/CMS/Deep Learning/expectation-maximization-algorithm/Dataset/model_gt.jpg"
        
        segmented_image = segment_image(image_path, expected_path, save_plot=True)
        print("Segmentation completed successfully. Results saved as 'segmentation_result.png'")
    except Exception as e:
        print(f"Program failed: {str(e)}")