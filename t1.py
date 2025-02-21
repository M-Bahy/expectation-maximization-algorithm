import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from random import choice
from math import pi, sqrt, exp

use_sklearn = False

def det_matrix(matrix):
    """Calculate determinant of a 2D matrix"""
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for i in range(len(matrix)):
        minor = [row[:i] + row[i+1:] for row in matrix[1:]]
        det += ((-1) ** i) * matrix[0][i] * det_matrix(minor)
    return det

def inverse_matrix(matrix):
    """Calculate inverse of a matrix"""
    n = len(matrix)
    # Create augmented matrix [matrix | I]
    aug = [[matrix[i][j] for j in range(n)] + [1 if i == j else 0 for j in range(n)] for i in range(n)]
    
    # Gaussian elimination
    for i in range(n):
        pivot = aug[i][i]
        for j in range(2*n):
            aug[i][j] = aug[i][j] / pivot
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2*n):
                    aug[k][j] -= factor * aug[i][j]
    
    # Extract right half (inverse matrix)
    return [[aug[i][j+n] for j in range(n)] for i in range(n)]

def create_identity_matrix(size):
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]

class GaussianMixtureModel_ByHand:
    def __init__(self, n_components=2, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter

    def initialize_parameters(self, X):
        """Initialize the model parameters"""
        n_samples = len(X)
        n_features = len(X[0])
        
        indices = list(range(n_samples))
        random_idx = [choice(indices) for _ in range(self.n_components)]
        self.means = [X[idx] for idx in random_idx]
        
        # Initialize covariance matrices as identity matrices
        self.covs = [create_identity_matrix(n_features) for _ in range(self.n_components)]
        
        self.mixing_coefficients = [1.0/self.n_components] * self.n_components
        
    def e_step(self, X):
        """Expectation step: compute responsibilities"""
        n_samples = len(X)
        # Initialize responsibilities as list of lists
        resp = [[0.0] * self.n_components for _ in range(n_samples)]
        
        # Compute probabilities for each component
        for k in range(self.n_components):
            probs = self.gaussian_pdf(X, self.means[k], self.covs[k])
            for i in range(n_samples):
                resp[i][k] = self.mixing_coefficients[k] * probs[i]
        
        # Normalize responsibilities
        for i in range(n_samples):
            row_sum = sum(resp[i])
            if row_sum == 0:
                row_sum = 1e-10  # Small constant for numerical stability
            resp[i] = [r / row_sum for r in resp[i]]
        
        return resp
    
    def gaussian_pdf(self, X, mean, cov):
        """Compute gaussian probability density function"""
        n_features = len(X[0])
        det = det_matrix(cov)
        if abs(det) < 1e-10:
            det = 1e-10  # Small constant for numerical stability
            
        norm_const = 1.0 / ((2 * pi) ** (n_features/2) * sqrt(abs(det)))
        inv_cov = inverse_matrix(cov)
        
        result = []
        for x in X:
            # Normalize x and mean to prevent overflow
            x = [float(val) / 255.0 for val in x]  # Assuming 8-bit image values
            mean = [float(val) / 255.0 for val in mean]
            
            x_mu = [x[i] - mean[i] for i in range(n_features)]
            # Calculate (x-μ)ᵀΣ⁻¹(x-μ)
            temp = [sum(a * b for a, b in zip(x_mu, col)) for col in zip(*inv_cov)]
            exp_term = -0.5 * sum(a * b for a, b in zip(temp, x_mu))
            
            # Prevent underflow in exp
            if exp_term < -700:  # log(min float)
                exp_term = -700
            result.append(float(norm_const * exp(exp_term)))
        
        return result
    
    def m_step(self, X, resp):
        """Maximization step: update parameters"""
        n_samples = len(X)
        n_features = len(X[0])
        
        # Calculate nk (sum of responsibilities for each component)
        nk = [sum(resp[i][k] for i in range(n_samples)) for k in range(self.n_components)]
        
        # Update mixing coefficients
        self.mixing_coefficients = [n/n_samples for n in nk]
        
        # Update means
        self.means = [[0.0] * n_features for _ in range(self.n_components)]
        for k in range(self.n_components):
            for j in range(n_features):
                self.means[k][j] = sum(resp[i][k] * X[i][j] for i in range(n_samples)) / nk[k]
        
        # Initialize covariance matrices properly
        self.covs = [[[0.0] * n_features for _ in range(n_features)] for _ in range(self.n_components)]
        
        # Update covariances
        for k in range(self.n_components):
            for i in range(n_features):
                for j in range(n_features):
                    sum_cov = 0.0
                    for n in range(n_samples):
                        diff_i = X[n][i] - self.means[k][i]
                        diff_j = X[n][j] - self.means[k][j]
                        sum_cov += resp[n][k] * diff_i * diff_j
                    self.covs[k][i][j] = sum_cov / nk[k]
            
            # Add small value to diagonal for numerical stability
            for i in range(n_features):
                self.covs[k][i][i] += 1e-6
    
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
        # Find index of max value for each sample
        return [max(range(len(r)), key=lambda i: r[i]) for r in resp]


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
    original_image_path = "/media/bahy/MEDO BAHY/CMS/Deep Learning/expectation-maximization-algorithm/Dataset/model.jpg"
    GMM(original_image_path)