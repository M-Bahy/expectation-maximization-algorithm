import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


use_sklearn = False

class GaussianMixtureModel_ByHand:
    def __init__(self, k=2, max_iter=10):
        self.k = k  # number of components
        self.max_iter = max_iter
        
    def compute_variance(self, X):
        """Compute variance manually for each feature"""
        mean = sum(X) / len(X)
        squared_diff = [(x - mean)**2 for x in X]
        variance = sum(squared_diff) / len(X)
        return variance
        
    def initialize_parameters(self, X):
        """Initialize model parameters manually"""
        n_samples, n_features = X.shape
        
        # Randomly choose k points as initial means
        random_indices = []
        while len(random_indices) < self.k:
            idx = int(np.random.random() * n_samples)
            if idx not in random_indices:
                random_indices.append(idx)
        self.means = X[random_indices]
        
        # Initialize variances using dataset variance
        variances = []
        for feature in range(n_features):
            feature_variance = self.compute_variance(X[:, feature])
            variances.append(feature_variance)
        self.variances = np.array([variances for _ in range(self.k)])
        
        # Initialize mixing coefficients equally
        self.mixing_coefficients = np.array([1.0/self.k] * self.k)
        
    def gaussian_pdf(self, x, mean, variance):
        """Compute gaussian probability density manually"""
        n_features = len(x)
        
        # Add small number to avoid division by zero
        variance = variance + 1e-6
        
        # Compute normalization constant
        norm_const = 1.0
        for std in np.sqrt(variance):
            norm_const *= 1.0 / (np.sqrt(2 * np.pi) * std)
        
        # Compute exponential term
        exp_term = 0
        for i in range(n_features):
            exp_term += ((x[i] - mean[i])**2) / (2 * variance[i])
        
        return norm_const * np.exp(-exp_term)
    
    def e_step(self, X):
        """Expectation step: compute responsibilities"""
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.k))
        
        # For each data point
        for i in range(n_samples):
            # Compute probability for each component
            for j in range(self.k):
                resp[i,j] = self.mixing_coefficients[j] * \
                           self.gaussian_pdf(X[i], self.means[j], self.variances[j])
            
            # Normalize to get responsibilities
            total = sum(resp[i])
            if total > 0:
                resp[i] = resp[i] / total
            else:
                resp[i] = np.array([1.0/self.k] * self.k)
                
        return resp
    
    def m_step(self, X, resp):
        """Maximization step: update parameters"""
        n_samples, n_features = X.shape
        
        # Update mixing coefficients
        nk = np.sum(resp, axis=0)
        self.mixing_coefficients = nk / n_samples
        
        # Update means
        for j in range(self.k):
            numerator = np.zeros(n_features)
            for i in range(n_samples):
                numerator += resp[i,j] * X[i]
            self.means[j] = numerator / nk[j]
        
        # Update variances
        for j in range(self.k):
            numerator = np.zeros(n_features)
            for i in range(n_samples):
                diff = X[i] - self.means[j]
                numerator += resp[i,j] * (diff ** 2)
            self.variances[j] = numerator / nk[j]
    
    def fit(self, X):
        """Fit the model to data"""
        self.initialize_parameters(X)
        
        for i in range(self.max_iter):
            print("Iteration", i+1)
            # E-step
            resp = self.e_step(X)
            # M-step
            self.m_step(X, resp)
        
        return self
    
    def predict(self, X):
        print("Predicting ...")
        """Predict cluster assignments"""
        resp = self.e_step(X)
        labels = []
        for r in resp:
            max_prob = max(r)
            labels.append(list(r).index(max_prob))
        return np.array(labels)




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
        gmm = GaussianMixtureModel_ByHand()
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
    original_image_path = "/media/bahy/MEDO BAHY/CMS/Deep Learning/expectation-maximization-algorithm/Dataset/model.jpg"
    GMM(original_image_path)