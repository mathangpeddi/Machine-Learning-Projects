# Logistic Regression-here the classifier is a linear classifier and so the prediction boundayr is a linear one and not a non-linear one

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values  #Based on the age and estimated salary we try to predict if the person has purchased the SUV or not
y = dataset.iloc[:, 4].values #So here age and estimated salary are the independent variables so indexes are 2,3 and  purchased is the dependent variable(index is 4)
#So 1 means that the user has bought the car and 0 means he hasn't bought the car

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) #300 obs in the training set and 100obs in the test set

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression  #import the right class and create object of that class
classifier = LogisticRegression(random_state = 0)  #here classifier is the object,we always use the randon_state parameter to get the same results
classifier.fit(X_train, y_train)   #then fit it to our training set

# Predicting the Test set results
y_pred = classifier.predict(X_test)  #this gives the prediction for each of the test set observeration

# Making the Confusion Matrix-to check the no of correct and incorrect predictions done by our logistic regression classifier
from sklearn.metrics import confusion_matrix  #here if it starts with small letter then its the function name(like confusion-small letter so its a function) 
cm = confusion_matrix(y_test, y_pred)   #but Logistic,LinearRegression all these start with the capital letter so they are classes,confusion is the function and not the class
#this means that there are 65correct predictions of 0,8 wrong predictions of 0 and 24correct predictions of 1 so total 89 correct predictions and 11 wrong predictions so 89% accuracy

# Visualising the Training set results
from matplotlib.colors import ListedColormap #this is for colourizing all the data points
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()  #now our classifer-based on the salary and age of the person predicts if he'll buy the SUV or not

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show() #We got 11 incorrect predictions here so 8 incorrect predictions for 0-thats why we have exactly 8 greendots in the red region
#3 incorrect predictions for 1 so thats why 3 red dots in the green region
