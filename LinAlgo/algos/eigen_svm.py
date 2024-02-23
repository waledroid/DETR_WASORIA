# eigen_svm.py
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Import PCACompression from eigen_VV.py here, if needed
from algos.eigen_VV import PCACompression

class SVMClassifier:
    def __init__(self):
        self.model = SVC(kernel='linear')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy}")

    def show_sample_results(self, X_original, X_test, y_test, num_samples=3):
        """
        Show sample original images with their predicted and actual labels.
        X_original: The original images before PCA reduction.
        X_test: The PCA-reduced test set (used only for predictions, not visualization).
        y_test: The true labels for the test set.
        """
        predictions = self.predict(X_test)  # Predict using the PCA-reduced features
        for i in range(num_samples):
            plt.imshow(X_original[i].reshape(28, 28), cmap='gray')  # Visualize the original image
            plt.title(f"Predicted: {predictions[i]}, Actual: {y_test[i]}")
            plt.show()
