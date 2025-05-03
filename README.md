# Machine-Learning-on-Blood-Pressure-Prediction

## Analysis Questions
### Task 1
1. Blood pressure dataset pressure.csv contains examples of systolic pressures and
other features from different persons.
2. Create polynomial regression models, to predict systolic pressure using the SERUMCHOL feature, for degrees vary from 1 to 14. Perform 10-fold cross validation.
Calculate its square roots of the mean square errors (RMSE), and the mean RMSE.
Display the mean RMSEs for the 14 different degrees. Produce a cross validation
error plot using the mean RMSE with 1 to 14 different degrees.
3. Select the best degree, and explains why briefly. Print its intercept and coefficients.
4. Create a multiple linear regression model to predict systolic pressure using all the
relevant features. Print its coefficients. Perform 10-fold cross validation. Calculate
its square roots of the mean square errors (RMSE), and the mean RMSE, and display
the mean RMSE.
5. Build a ridge regression model of the above (i.e. item 4) using α = 0.1. Print its
coefficients. Perform 10-fold cross validation. Calculate its square roots of the mean
square errors (RMSE), and the mean RMSE, and display the mean RMSE.
6. Select the best model of the three, and explains why briefly

### Task 2
1. This task involves MNIST Digit Classification using PCA and Logistic Regression.
Load the renowned MNIST (’mnist 784’) dataset, which consists of a large collection
of handwritten digit images. Your task is to reduce the number of features first, and
then build a binary classification model to distinguish between the digit “6” and all
other digits (not “6”).
2. Perform Principal Component Analysis (PCA) on the feature data to reduce its dimensionality while retaining 88% of the overall explained variance ratio.
3. Split the data into training and testing sets. A common split ratio is 80% training
and 20% testing.
4. Create a Logistic Regression model using the reduced feature dataset.
5. Use this model to predict the language for the training dataset and the testing
dataset.
6. Print the number of principal components preserved. Print the prediction accuracy
(proportion of correct predictions) of your model on the training set. Print the
prediction accuracy, the confusion matrix, and the misclassified digits (i.e. wrong
predicitons) of your model on the testing set.
7. What do you think of the model generated (good, underfit, overfit)? Briefly explains
why


## Library:
- ML: sklearn
- Data Visualisation: matplotlib
- Dataframe Manipulation: pandas, numpy, 

## Methods:
### Task 1:
- Polynomial regression, Multiple Linear Regression, Ridge regression
- K-fold Cross-Validation 
- Optimal Model Selection

### Task 2:
- Training and testing data split
- PCA (Principal Component Analysis)
- Logistic Regression
- Model predictions and misclassification prediction.****
