# Ridge Regression and Lasso Regression-these 2 algorithms are the regularization algorithms and they are used to avoid overfitting.
# By these algorithms we add an extra term to the Least Squares and try to reduce the variance. So after applying this model the variance decreases but the bias increases.

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
from sklearn.datasets import load_boston # Here we load the inbuild dataset which is the Boston dataset
df=load_boston()
df

dataset = pd.DataFrame(df.data)
print(dataset.head())

dataset.columns=df.feature_names
dataset.head()

df.target.shape
dataset["Price"]=df.target
dataset.head()

X=dataset.iloc[:,:-1] # Independent features
y=dataset.iloc[:,-1] # Dependent features

# Linear Regression- here I have included Linear Regression just to compare the results
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression # Linear Regression model

lin_regressor=LinearRegression()
mse=cross_val_score(lin_regressor,X,y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)

# Ridge Regression
from sklearn.linear_model import Ridge # Ridge Regression model
from sklearn.model_selection import GridSearchCV # We use the GridSearchCV to get the best value of alpha

# Fitting the Ridge Regression to the Training Set
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5) # cv indicates Cross validation,so this is equal to 5
ridge_regressor.fit(X,y)

print(ridge_regressor.best_params_) # It will return the value of alpha corresponding to the best score-this returns alpha:100
print(ridge_regressor.best_score_) #  The model which is closer to 0 or the model which has a greater score is a better model. So we compare all the 3 models and then come to a conclusion

# Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

# Fitting the Lasso Regression to the Training Set
lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_) # It will return the value of alpha corresponding to the best score-this returns alpha:1
print(lasso_regressor.best_score_)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Predicting the Test set results
prediction_lasso=lasso_regressor.predict(X_test)
prediction_ridge=ridge_regressor.predict(X_test)

# Visualizing Lasso Regression
import seaborn as sns
sns.distplot(y_test-prediction_lasso)

# Visualizing Ridge Regression
sns.distplot(y_test-prediction_ridge)

# After comparing all the 3 models we find that the Ridge Regression model is the best fit as the score is high compared to other models.
# But for large datasets the Lasso Regression performs well as it ignores the featues which are not of much importance. This takes care of feature selection.
# But as a whole, both these regression techniques are better than Linear Regression as they increase the bias by a small margin but decrease the variance by avoiding overfitting.