##Task 2##
#1. This task involves MNIST Digit Classification using PCA and Logistic Regression. Load the renowned 
#MNIST (’mnist 784’) dataset, which consists of a large collection of handwritten digit images. Your task is 
#to reduce the number of features first, and then build a binary classification model to distinguish between 
#the digit “6” and all other digits (not “6”).


from sklearn.datasets import fetch_openml 
mnist = fetch_openml("mnist_784", version=1, as_frame=False) #Downloading MNIST data set to reduce dimensions.

import numpy as np
import pandas as pd
mnistdf = pd.DataFrame(data=mnist.data, columns=mnist.feature_names)
y = mnistdf['target'] = np.where(mnist.target.astype(int) == 6, 1, 0)

X = mnist.data
y 
# Created a new X and y variables to do Logistics, PCA and train-test split with 6, 1 and 0 in the MNIST data set.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
Xtrain_scale =  scaler.fit_transform(X_train)
Xtest_scale = scaler.transform(X_test)

#2. Perform Principal Component Analysis (PCA) on the feature data to reduce its dimensionality while retaining 88% of the overall 
#explained variance ratio.

from sklearn.decomposition import PCA
ratio = 0.88 #setting the n_component as ratio as 0.88
pca = PCA(n_components=ratio, svd_solver='full')
Xpca = pca.fit_transform(X)
selectedcomp = pca.n_components_

#PCA or Principal Component Analysis, finds the hyperplane that lies closest to the dataset and
# project the dataset onto the hyperplane.

#3. Split the data into training and testing sets. A common split ratio is 80% training and 20% testing.

X_train2, X_test2, y_train2, y_test2 = train_test_split(Xpca, y, test_size=0.2, random_state=42)
print(f"X_train = {X_train2}")
print(f"X_test: {X_test2}")
print(f"y_train: {y_train2}")
print(f"y_test {y_test2}")

#I have the train test split to find the accuracy of the PCA.

# 4. Create a Logistic Regression model using the reduced feature dataset.

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', max_iter=5000)
model.fit(X_train2, y_train2)

#I put the logistic regression to return all the 0s and 1s.

# 5. Use this model to predict the language for the training dataset and the testing dataset.

ytrain_predict = model.predict(X_train2)
ytest_predict = model.predict(X_test2)

print(f"The prediction is {ytest_predict}")
print(f"The prediction is {ytrain_predict}")

#Using predict function to find the prediction of the train and test sets.

#6. Print the number of principal components preserved. Print the prediction accuracy (proportion of correct predictions) of your model on 
#the training set. Print the prediction accuracy, the confusion matrix, and the misclassified digits (i.e. wrong predicitons) of your model 
#on the testing set.

from sklearn.metrics import accuracy_score
accuracytest = accuracy_score(y_test, ytest_predict)
accuracytrain = accuracy_score(y_train, ytrain_predict)

from sklearn.metrics import confusion_matrix
confused = confusion_matrix(y_test, ytest_predict)
misclassified = np.where(y_test !=ytest_predict)[0]


from sklearn.metrics import classification_report
classificationtest = classification_report(y_test, ytest_predict)
classificationtrain = classification_report(y_train, ytrain_predict)

print(f"The Accuracy test is: {accuracytest}")
print(f"The Accuracy train is: {accuracytrain}")
print(f"The Confusion Matrix is: {confused}")
print(f"The Classification Report is: {classificationtest}")
print(f"Number of Misclassified Digits: {misclassified}")
