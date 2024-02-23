This Package performs 3 separate Linear Algorithms on MNIST Datatset:<br>
### 1 - Singular Value Decomposition <br>
### 2 - The calculation of Eigen-values and Eigen-vector <br>
### 3 - Linear Regression. <br>

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
 # svm     -- for Support vector Machine to classify the reduced data from PCA
 # linereg -- for classification using linear Regression on MNIST dataset  
</pre>  
 
#### SVD:
Applying SVD reduces the dimensionality of the data while preserving its most significant features.

 In the SVD, the original image is decomposed into three matrices 
