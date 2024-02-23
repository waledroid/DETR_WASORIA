import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class PCACompression:
    def __init__(self, n_components=0.95):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.eigenvectors = None
        self.n_components_selected = None

    def fit_transform(self, data):
        """Fit the PCA model on the training data and transform it."""
        # Standardize the data
        standardized_data = self.scaler.fit_transform(data)
        
        # Compute covariance matrix
        cov_matrix = np.cov(standardized_data, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select the number of components
        if self.n_components < 1:
            cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            self.n_components_selected = np.argmax(cumulative_variance >= self.n_components) + 1
        else:
            self.n_components_selected = self.n_components
        
        # Reduce dimensions
        reduced_data = np.dot(standardized_data, eigenvectors[:, :self.n_components_selected])
        self.eigenvectors = eigenvectors
        return reduced_data

    def transform(self, data):
        """Transform the test data using the eigenvectors obtained from the training set."""
        # Standardize the data using the same scaler as the training set
        standardized_data = self.scaler.transform(data)
        # Reduce dimensions using the stored eigenvectors
        reduced_data = np.dot(standardized_data, self.eigenvectors[:, :self.n_components_selected])
        return reduced_data


