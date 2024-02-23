import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class Linereg_MNIST:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        """Train the Linear Regression model."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions using the trained model."""
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """Evaluate the model and print MSE and accuracy."""
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        
        y_pred_rounded = np.round(y_pred)
        accuracy = np.sum(y_pred_rounded == y_test) / len(y_test)
        accuracy_percentage = accuracy * 100
        print(f"Accuracy (Rounded Predictions): {accuracy_percentage:.2f}%")
        return accuracy_percentage

    def show_sample_predictions(self, X_test_original, y_test, num_samples=3):
        """Display sample images with their actual and predicted labels."""
        y_pred = self.predict(X_test_original.reshape(X_test_original.shape[0], -1))
        for i in range(num_samples):
            plt.figure(figsize=(2, 2))
            plt.imshow(X_test_original[i].reshape(28, 28), cmap='gray')
            plt.title(f"Predicted: {round(y_pred[i])}, Actual: {y_test[i]}")
            plt.axis('off')
            plt.show()
