import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


use_sklearn = False

class GaussianMixtureModel_ByHand:
    def __init__(self, n_components=2, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter
        
    def initialize_parameters(self, X):
        """Initialize the model parameters"""
        n_samples, n_features = X.shape
        
        # Randomly initialize means by selecting a random sample from the data as the mean
        random_idx = np.random.permutation(n_samples)[:self.n_components]
        self.means = X[random_idx]
        
        # Initialize covariances
        self.covs = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
        # Initialize mixing coefficients (mixing_coefficients)
        self.mixing_coefficients = np.ones(self.n_components) / self.n_components
        
    def gaussian_pdf(self, X, mean, cov):
        """Compute gaussian probability density function"""
        n_features = X.shape[1]
        det = np.linalg.det(cov)
        if det == 0:
            det = np.finfo(float).eps
            
        norm_const = 1.0 / (np.power(2 * np.pi, n_features/2) * np.sqrt(det))
        inv_cov = np.linalg.inv(cov)
        
        X_mu = X - mean
        exp = -0.5 * np.sum(X_mu.dot(inv_cov) * X_mu, axis=1)
        
        return norm_const * np.exp(exp)
    
    def e_step(self, X):
        """Expectation step: compute responsibilities"""
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.n_components))
        
        # Compute probabilities for each component
        for k in range(self.n_components):
            resp[:, k] = self.mixing_coefficients[k] * self.gaussian_pdf(X, self.means[k], self.covs[k])
            
        # Normalize responsibilities
        resp_sum = resp.sum(axis=1)[:, np.newaxis]
        resp_sum[resp_sum == 0] = np.finfo(float).eps
        resp = resp / resp_sum
        
        return resp
    
    def m_step(self, X, resp):
        """Maximization step: update parameters"""
        n_samples = X.shape[0]
        
        # Update mixing_coefficients
        nk = resp.sum(axis=0)
        self.mixing_coefficients = nk / n_samples
        
        # Update means
        self.means = resp.T.dot(X) / nk[:, np.newaxis]
        
        # Update covariances
        for k in range(self.n_components):
            X_mu = X - self.means[k]
            self.covs[k] = (X_mu.T * resp[:, k]).dot(X_mu) / nk[k]
            
            # Add small value to diagonal for numerical stability
            self.covs[k].flat[::X.shape[1] + 1] += 1e-6
    
    def fit(self, X):
        """Fit the model to the data"""
        # Initialize parameters
        self.initialize_parameters(X)
        
        for iteration in range(self.max_iter):
            # E-step
            resp = self.e_step(X)
            
            # M-step
            self.m_step(X, resp)
            
        return self
    
    def predict(self, X):
        """Predict cluster labels"""
        resp = self.e_step(X)
        return resp.argmax(axis=1)




def GMM (original_image_path):

    image = Image.open(original_image_path)
    image_array = np.array(image)
    height, width, channels = image_array.shape
    # convert the image into pixels 2D array as expected by the model
    pixels = image_array.reshape(-1, channels)

    # 2 components foreground and background so the return is either 0 or 1
    # takes a 2D array were each row is a pixel and each column is a feature (color channel)
    if use_sklearn:
        print("Using sklearn ...")
        gmm = GaussianMixture(n_components=2)
    else:
        print("Using custom implementation ...")
        gmm = GaussianMixtureModel_ByHand(n_components=2)
    gmm.fit(pixels)
    labels = gmm.predict(pixels)

    if np.sum(labels) < len(labels) / 2:
        foreground_label = 1
    else:
        foreground_label = 0

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
    original_image_path = "/media/bahy/MEDO BAHY/CMS/Deep Learning/expectation-maximization-algorithm/Dataset/by_hand.jpg"
    GMM(original_image_path)