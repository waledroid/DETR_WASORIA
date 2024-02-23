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
 # svm     -- for Support vector Machine to classify the reduced data from PCA
 # linereg -- for classification using linear Regression on MNIST dataset  
</pre>  
 
#### SVD:
We pass a batch (64) from MNIST,<br> 
for every image in the batch, Apply SVD to reduce the dimensionality of the data while preserving its most significant features.

 In the SVD, the original image is decomposed into three matrices 
<br>
 <img src="../images/svd.png" width="500" >
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

