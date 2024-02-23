import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

class SVDCompression:
    def compress_and_reconstruct(self, dataloader):
        for images, _ in dataloader:
            # Use one batch for demonstration
            print(f"Number of images in the batch: {images.shape[0]}")
            batch_size = images.shape[0]
            
            # Ensure the output directory exists
            if not os.path.exists('SVD_reconstructed_images/'):
                os.makedirs('SVD_reconstructed_images/')
            
            for i in range(batch_size):
                # Flatten and center the image by subtracting its mean
                image = images[i].numpy().squeeze()
                image_centered = image - np.mean(image)  # Centering the image

                # Apply SVD on the centered image
                U, s, Vt = np.linalg.svd(image_centered, full_matrices=False)
                k = len(s) // 2  # Keeping top 50% singular values
                # k = 9  # Specify the number of singular values to keep

                # Create the reduced diagonal matrix of singular values
                S_k = np.diag(s[:k])

                # Reduce U and V^T according to k
                U_k = U[:, :k]
                Vt_k = Vt[:k, :]

                # Reconstruct the image from the reduced components
                reconstructed = np.dot(U_k, np.dot(S_k, Vt_k)) + np.mean(image)  # Add the mean back to reconstruct

                # Save each reconstructed image
                plt.imsave(f'SVD_reconstructed_images/reconstructed_{i}.png', reconstructed, cmap='gray')
                
                # For demonstration, plot one original and its SVD reconstructed image
                if i == 0:  # This plots the comparison for the first image in the batch
                    plt.subplot(1, 2, 1)
                    plt.imshow(image, cmap='gray')
                    plt.title('Original')
                    plt.subplot(1, 2, 2)
                    plt.imshow(reconstructed, cmap='gray')
                    plt.title('SVD Reconstructed')
                    plt.show()
                
            break  # Process only the first batch for demonstration