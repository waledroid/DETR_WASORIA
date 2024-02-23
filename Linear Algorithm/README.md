This Package performs 3 separate Linear Algorithms on MNIST Datatset:<br>
#### 1 - Singular Value Decomposition <br>
#### 2 - The calculation of Eigen-values and Eigen-vector <br>
#### 3 - Linear Regression. <br>

Below is the file structure designed:
<pre>
LinAlgo/
├── __init__.py
├── main.py (call functions in /algos/ )
└── algos/ 
    ├── __init__.py
    ├── svd.py
    |── eigen_VV.py
    └── linear_regression.py
    └── SVD reconstructed_images/
</pre>

 <br>
Run the main and make sure to parse other arguments, example:  
<pre>
 python main.py svd
 # svd     -- for Singular Value Decomposition 
 # pca     -- for Eigen Vector, Eigen Valuee  
 # svm     -- SVM for PCA-Transformed MNIST Classification
 # linereg -- Linear Regression for Digit Recognition on MNIST dataset  
</pre>  
 
#### SVD:
We pass a batch (64) from MNIST,<br> 
for every image in the batch, Apply SVD to reduce the dimensionality of the data while preserving its most significant features.

 In the SVD, the original image is decomposed into three matrices 
<br>
 <img src="images/svd.png" width="500" >
<br>
The matrix S in the full SVD is a diagonal matrix of singular values, reduce it to keep only the top 
k singular values (for compression), reduce also the dimensions of U and V transpose

##### How
<pre>
1. Select the top k singular values in S: 
2. Reduce U to U_k by keeping only the columns corresponding to the top k singular values.
3. Reduce V transpose to V_k Transpose by keeping only the rows corresponding to the top k singular values.
4. Reconstruct the image using the reduced matrices: 
U_k⋅S_k⋅VkT, where S k is the diagonal matrix containing only the top k singular values
</pre>


<br>
<br>

### Principal Component Analysis (PCA):
PCA provides both the directions and the magnitude of the variance of the data in the new feature space, enabling dimensionality reduction while preserving as much of the data's original structure as possible.
##### How
<pre>
We start by calculating the covariance matrix of the data set, to understand variation between data. 
Then compute the eigenvalues (directions) and eigenvectors (magnitude of variance) of this covariance matrix
</pre>
<br>

### SVM for PCA-Transformed MNIST Classification:
For classification, the main idea behind SVM is to find the hyperplane that best separates the classes in the feature space. In a two-dimensional space, this hyperplane is a line, but in higher dimensions, it's a plane or a hyperplane.

The best hyperplane is the one that has the maximum margin, which is the maximum distance between the hyperplane and the nearest data point from either class. These nearest data points are called support vectors, as they support or define the hyperplane.
##### How
<pre>
since MNIST is a multiclass classification problem, use the SVC class from the sklearn.svm module. SVC will automatically use one-vs-one or one-vs-all strategy for multiclass classification.
- Determine linear kernel or non-linear kernel (e.g., RBF, polynomial) based on the complexity of the transformed features using simple grid search with cross-validation.
- Train the SVM Classifier: bFit method of the SVC class with the PCA-reduced features as input and the corresponding labels.
- Evaluate the Model:
- Use the trained SVM model to predict the classes of the PCA-transformed test set. Assess the model's performance using appropriate metrics, such as accuracy, precision, recall, or the confusion matrix. 
</pre>

### Linear Regression for Digit Recognition:
While linear Regression for classification is unconventional, The objective of linear regression is to minimize the sum of the squared residuals (differences between observed and predicted values), known as the least squares criterion
