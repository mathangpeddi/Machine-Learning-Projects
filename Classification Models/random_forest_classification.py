# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0) #n_estimators-the no of trees in the forest
classifier.fit(X_train, y_train) #for each user we will have 10 trees whether he will buy the SUV or not
#We shud also be able to detect overfitting,overfitting happens when we use most of the part for training set only,so it fits the training set data perfectly.
#But if we want to test for new values suppose if we want to try that for the test set then the model will be lost and wont be good,it'll be used for the training set only and wont be able to fit the new examples properly

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) #so here we get 9 incorrect predictions,4 for 0 and 5 for 1 so 91% accuracy

# Visualising the Training set results
from matplotlib.colors import ListedColormap
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
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
#The random forest calculates the majority of votes for 0 or 1,it takes the value from 10 trees,the user's prediction is done for 10 trees here
#Suppose if red region(0) has 3 votes and green region(1) has 7 votes then majority of votes is for green region so the random forest predicts it as 1(green region)
#So accordingly the random forest takes the predictions from one user(10 predictions) and whichever has the majority of votes the data point(which corresponds to user)belongs to that region

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
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
#Here we can easily find overfitting because when we observe the training set the irregularities are very high
#For eg.the right side contains the red rectangles for the red data points in the training set,but when it comes to the test set(new examples)there are no red data points,the red region is fitted for the red points which leads to overfitting
#Out of all the best classifier is the kernal SVM because it was a smooth curve and it properly differentiated the red and the green points,linear classifier was not so good,the random forest and decision tree are having many irregularities,so considering the accuracy and the plot the kernel SVM was the best fit/classifier for this business problem.
#Even the Naive Bayes classifier had a smooth curve prediction boundary which put most of the red data points in the correct red region.
 