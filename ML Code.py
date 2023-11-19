# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define auxiliary function to plot the confusion matrix
def plot_confusion_matrix(y, y_predict):
    cm = confusion_matrix(y, y_predict)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['did not land', 'land'])
    ax.yaxis.set_ticklabels(['did not land', 'land'])
    plt.show()

# TASK 1: Create a NumPy array from the column Class in data
# Y = data['Class'].to_numpy()

# TASK 2: Standardize the data in X
# Assuming X is loaded with the appropriate data
# transform = preprocessing.StandardScaler()
# X = transform.fit_transform(X)

# TASK 3: Split the data into training and test data
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# TASK 4: Create a logistic regression object and perform grid search
# parameters_lr = {'C': [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}
# lr = LogisticRegression()
# logreg_cv = GridSearchCV(lr, parameters_lr, cv=10)
# logreg_cv.fit(X_train, Y_train)

# TASK 5: Calculate the accuracy on the test data
# accuracy_lr = logreg_cv.score(X_test, Y_test)

# TASK 6: Create a support vector machine object and perform grid search
# parameters_svm = {
#     'kernel': ('linear', 'rbf', 'poly', 'rbf', 'sigmoid'),
#     'C': np.logspace(-3, 3, 5),
#     'gamma': np.logspace(-3, 3, 5)
# }
# svm = SVC()
# svm_cv = GridSearchCV(svm, parameters_svm, cv=10)
# svm_cv.fit(X_train, Y_train)

# TASK 7: Calculate the accuracy on the test data
# accuracy_svm = svm_cv.score(X_test, Y_test)

# TASK 8: Create a decision tree classifier object and perform grid search
# parameters_tree = {
#     'criterion': ['gini', 'entropy'],
#     'splitter': ['best', 'random'],
#     'max_depth': [2 * n for n in range(1, 10)],
#     'max_features': ['auto', 'sqrt'],
#     'min_samples_leaf': [1, 2, 4],
#     'min_samples_split': [2, 5, 10]
# }
# tree = DecisionTreeClassifier()
# tree_cv = GridSearchCV(tree, parameters_tree, cv=10)
# tree_cv.fit(X_train, Y_train)

# TASK 9: Calculate the accuracy on the test data
# accuracy_tree = tree_cv.score(X_test, Y_test)

# TASK 10: Create a k nearest neighbors object and perform grid search
# parameters_knn = {
#     'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     'p': [1, 2]
# }
# knn = KNeighborsClassifier()
# knn_cv = GridSearchCV(knn, parameters_knn, cv=10)
# knn_cv.fit(X_train, Y_train)

# TASK 11: Calculate the accuracy on the test data
# accuracy_knn = knn_cv.score(X_test, Y_test)

# TASK 12: Find the method that performs best
# accuracies = {
#     'Logistic Regression': accuracy_lr,
#     'Support Vector Machine': accuracy