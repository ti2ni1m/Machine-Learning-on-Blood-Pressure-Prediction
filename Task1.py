##Task 1##
#1. Blood pressure dataset pressure.csv contains examples of systolic pressures and other features from 
#different persons.
import pandas as pd
import numpy as np

blood = pd.read_csv("bloodpressure-23.csv") #Reading the csv file here
print(blood) #Printing out all the variables

#2. Create polynomial regression models, to predict systolic pressure using the SERUMCHOL feature, for degrees 
# vary from 1 to 14. Perform 10-fold cross validation. Calculate its square roots of the mean square errors 
# (RMSE), and the mean RMSE. Display the mean RMSEs for the 14 different degrees. Produce a cross validation 
# error plot using the mean RMSE with 1 to 14 different degrees.

X = blood[['SERUM-CHOL']] #Assigining X as 'SERUM-CHOL' from the blood pressure data
y = blood[['SYSTOLIC']] #Assigining y as 'SYSTOLIC' from the blood pressure data

mean_rsme_value = []
rsme_value = []

degrees = np.arange(1, 15) #14 degrees here

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

for degree in degrees :
    polynomial = PolynomialFeatures(degree)
    X_polynomial = polynomial.fit_transform(X)
    linearmodel = LinearRegression()
    y_pred = cross_val_score(linearmodel, X_polynomial, y, cv=10, scoring = "neg_mean_squared_error")
    rsme =  np.mean(np.sqrt(-y_pred))
    rsme_value.append(rsme)
    mean_rsme_value.append(np.mean(rsme))
    print(f"The mean of Degree 14 is: {np.mean(rsme)}")
    #A for loop to create a polynomial regression which generalises linear 
    #regression by allowing exponents to occur in each feature for finding the mean of the RMSE.
    
#Note: I accidently named RMSE ad rsme variable.
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(degrees, mean_rsme_value, marker = 'o', linestyle = '-')
plt.title('Cross Valiation Error and Polynomial Regression Degree')
plt.xlabel('Polyniomial Degree')
plt.ylabel('Mean RSME')
plt.grid(True)
plt.show()

# Creating a plot to display the polynomial regression.

for degree, rsme in zip(degrees, mean_rsme_value) :
    print(f"Degree {degree} : Mean RSME = {rsme:.2f}")
    
print(f"Mean RMSE of Polynomial Regression: {mean_rsme_value}")


#3. Select the best degree, and explains why briefly. Print its intercept and coefficients.
 
bestdegree = degrees[np.argmin(rsme_value)]
bestpolynomial = PolynomialFeatures(degree=bestdegree)
X_bestpolynomial = bestpolynomial.fit_transform(X)
bestmodel = LinearRegression()
bestmodel.fit(X_bestpolynomial, y)
bestrsme = mean_rsme_value[bestdegree]

print(f"Coeffiecients for the Best Degree {bestdegree} Model is: {bestmodel.coef_}")
print(bestmodel.coef_)
print(bestmodel.intercept_)
print(f"The mean of the Best Degree which is {bestdegree} is: {bestrsme}")

#The best degree is 2 because it is optimal whislt the other degrees are ovrfitting.#

#4. Create a multiple linear regression model to predict systolic pressure using all the 
#relevant features. Print its coefficients. Perform 10-fold cross validation. Calculate 
#its square roots of the mean square errors (RMSE), and the mean RMSE, and display 
#the mean RMSE.

#I decided to create multiple X, y, and linearmodel variables is because I thought it would be easier to classify and separate just to make it look easier.
X2 = blood[['AGE', 'ED-LEVEL', 'SMOKING STATUS', 'EXERCISE', 'WEIGHT', 'SERUM-CHOL', 'IQ', 'SODIUM']]
y2 = blood['SYSTOLIC']

linearmodel2 = LinearRegression()

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
validationscore = cross_val_score(linearmodel2, X2, y2, cv=kfold, scoring='neg_mean_squared_error')
rsme2 = np.sqrt(-validationscore)

mean_rsme_value2 = np.mean(rsme2)

linearmodel2.fit(X2, y2)
coefficients = linearmodel2.coef_
intercept = linearmodel2.intercept_

#Creating the multiple linear regression which estimates the relationship between a 
#quantitative dependent variable and two or more independent variables using a straight line

print("Coefficients:")
for feature, coef in zip(X2.columns, coefficients):
    print(f"{feature}: {coefficients}")
print(f"Intercept: {intercept}")

print(f"Mean RMSE for Multiple Linear Regression: {mean_rsme_value2}")

#5. Build a ridge regression model of the above (i.e. item 4) 
#using α = 0.1. Print its coefficients. Perform 10-fold cross 
#validation. Calculate its square roots of the mean square 
#errors (RMSE), and the mean RMSE, and display the mean RMSE. 

from sklearn.linear_model import Ridge

X3 = blood[['AGE', 'ED-LEVEL', 'SMOKING STATUS', 'EXERCISE', 'WEIGHT', 'SERUM-CHOL', 'IQ', 'SODIUM']]
X4 = pd.get_dummies(X3, drop_first=True)
y3 = blood['SYSTOLIC']

ridgemodel = Ridge(alpha=0.1)
kfold2 = KFold(n_splits=10, shuffle=True, random_state=42)
validationscore2 = cross_val_score(ridgemodel, X4, y3, cv=kfold2, scoring='neg_mean_squared_error')
rsme3 = np.sqrt(-validationscore2)

mean_rsme_value3 = np.mean(rsme3)

ridgemodel.fit(X3, y3)
coefficients2 = ridgemodel.coef_
intercept2 = ridgemodel.intercept_

#Ridge regression which is a regualarlisde version of Linear Regression
#that prevents overfitting, but not very useful for data with huge dimensions

print("Coefficients:")
for feature, coef in zip(X3.columns, coefficients2):
    print(f"{feature}: {coefficients2}")
print(f"Intercept: {intercept2}")

print(f"Mean RMSE for Ridge Regression: {mean_rsme_value3}")






