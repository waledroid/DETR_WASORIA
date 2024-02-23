# main.py
import argparse
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from algos.svd import SVDCompression
from algos.eigen_VV import PCACompression
from algos.eigen_svm import SVMClassifier
from algos.linereg import Linereg_MNIST

def process_data_with_pca(trainloader, testloader, n_components=0.95):
    pca_compressor = PCACompression(n_components=n_components)
    train_images, train_labels = next(iter(trainloader))
    test_images, test_labels = next(iter(testloader))

    # Flatten the images for PCA transformation
    train_images_flattened = train_images.view(train_images.size(0), -1).numpy()
    test_images_flattened = test_images.view(test_images.size(0), -1).numpy()
    
    # Reduce train and test datasets
    reduced_train_data = pca_compressor.fit_transform(train_images_flattened)
    reduced_test_data = pca_compressor.transform(test_images_flattened)
    
    return reduced_train_data, train_labels.numpy(), reduced_test_data, test_labels.numpy(), train_images_flattened, test_images

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Linear Algebra Algorithms on MNIST')
    parser.add_argument('mode', type=str, help='svd for SVD, pca for PCA, svm for SVM classification, linreg for Linear Regression')
    args = parser.parse_args()

    # MNIST Data loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    if args.mode == 'svd':
        # SVD Compression and Reconstruction
        svd_compressor = SVDCompression()
        svd_compressor.compress_and_reconstruct(trainloader)

    elif args.mode == 'pca' or args.mode == 'svm':
        # Process data with PCA
        reduced_train_data, train_labels, reduced_test_data, test_labels, train_images_flattened, test_images = process_data_with_pca(trainloader, testloader)

        if args.mode == 'pca':
            # For demonstration, plot one sample image after PCA
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(train_images_flattened[0].reshape(28, 28), cmap='gray')
            plt.title('Original Image')
            plt.subplot(1, 2, 2)
            plt.imshow(reduced_train_data[0].reshape(1, -1), cmap='gray', aspect='auto')
            plt.title('After PCA')
            plt.show()

        elif args.mode == 'svm':
            # SVM classification
            svm_classifier = SVMClassifier()
            svm_classifier.train(reduced_train_data, train_labels)
            svm_classifier.evaluate(reduced_test_data, test_labels)

            # Visualization for SVM results requires original test images
            X_test_original = test_images.numpy()
            svm_classifier.show_sample_results(X_test_original, reduced_test_data, test_labels, num_samples=3)

    elif args.mode == 'linereg':
        # Linear Regression with CNN
        # Extract data from loaders
        train_images, train_labels = next(iter(trainloader))
        test_images, test_labels = next(iter(testloader))

        X_train = train_images.view(train_images.size(0), -1).numpy()
        y_train = train_labels.numpy()
        X_test = test_images.view(test_images.size(0), -1).numpy()
        y_test = test_labels.numpy()
        X_test_original = test_images.numpy()

        # Initialize, train, and evaluate the Linear Regression model
        lr_mnist = Linereg_MNIST()
        lr_mnist.train(X_train, y_train)
        lr_mnist.evaluate(X_test, y_test)

        # Show sample image predictions
        lr_mnist.show_sample_predictions(X_test_original, y_test, num_samples=3)
        pass

if __name__ == "__main__":
    main()