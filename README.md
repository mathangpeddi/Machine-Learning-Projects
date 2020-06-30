# Supervised Learning
   ## • Regression
   For the Simple Linear Regression model I have used the Salary dataset where I had to predict the Salary of a person based on his years of experience. For the Multiple Linear Regression model I have used the 50Startups dataset where I had to predict the Profit based on various features-which included numerical as well as categorical variables. Then I have taken the PositionSalaries dataset where I had to predict the Salary of a person based on his level. For the Ridge,Lasso Regression I have used the inbuilt dataset which is the Boston dataset where we try to predict the prices of houses in various locations in Boston. So the different regression algorithms used are:
   <ul>
     <li>Linear Regression</li>
    <ul style="list-style-type:circle">
         <li>Simple Linear Regression</li>
         <li>Multiple Linear Regression</li>
         <li>Polynomial Regression</li>
    </ul>
     <li>Support Vector Regression</li>
     <li>Decision Tree Regression</li>
     <li>Random Forest Regression</li>
     <li>Ridge Regression</li>
     <li>Lasso Regression</li>
   </ul>
   
   ## • Classification
 This is the Social Networks Ads dataset which consists of 400 obersvations corresponding to 400 columns. Each row is a customer and for each of them we take only the 2 features Age and estimated salary(as UserId and gender are not of much importance to us) with which we are going to predict this dependent variable which tells if he has purchased the car or not(yes or no). So first we take the dataset and split it into training and test sets and then train this model to understand the correlations between these features and the dependent variable vector we will be able to predict which new customer will buy that brand new SUV just released by this car company and therefore  make predictions on the test set and compare it with the actual values and calculate the accuracy. The different classification algorithms used are:
   <ul>
    <li>Logistic Regression</li>
    <li>K Nearest Neighbours (KNN)</li>
    <li>Support Vector Machine (SVM)</li>
    <li>Kernel SVM</li>
    <li>Naive Bayes</li>
    <li>Decision Tree Classification</li>
    <li>Random Forest Classification</li>
   </ul>
 Out of all the models,the best classifier is the kernel SVM because it showed a smooth curve and properly differentiated the red and the green regions. The linear classifier was not so good,the random forest and decision tree are having many irregularities. So considering the accuracy and the plot, the kernel SVM was the best fit/classifier for this business problem.
   
# Unsupervised Learning
This dataset is made by the strategic team of the mall and they have collected soe data about their customers.Each row corresponds to the customer of the mall having various attributes such as CustomerID,Gender,Age,Annual Income and Spending Score.Spending Score is a metric made by a mall to measure how uch each customer spends.Lower the score the lesser the customer spends and higher the score the more the customer spends and this takes values from 1 to 100. So the team wants to identify some patterns within its customers. So we need to create a dependent variable which contains the classes of all these customers.
   <ul>
     <li>Clustering</li>
    <ul style="list-style-type:circle">
         <li>K Means Clustering</li>
         <li>Hierarchial Clustering</li>
   </ul>
   </ul>


# Dimensionality Reduction
So here I have chosen the Wine dataset.There are 13 columns-independent variables and only one dependent variable,so the business owner-suppose if he takes a new obs then he shud be able to predict the customer segment(to which category he belongs to),the owner can recommend the new wine to the right customers.For each new wine it tells us to which customer segment it will be more appropriate.Here I have built a logistic regression model for prediction,but to see the visualizations there are 13 dimensions and they cannot be visualized. So we need to apply dimensionality reduction techniques and reduce the dataset to a lower number of features. Mainly we need to reduce the dimensions and provide a simple dataset which can give excellent correlations and which when applied to a logistic regression model gives accurate results.
<ul>
    <li>Principal Component Analysis</li>
   This is an unsupervised machine learning algorithm.
    <li>Linear Discriminant Analysis</li>
   This is a supervised machine learning algorithm.
</ul>
