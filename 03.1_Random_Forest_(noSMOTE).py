# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import seaborn as sns

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# +
import pandas as pd

X_train = pd.read_csv("X_train_processed.csv")

y_train = pd.Series.from_csv('y_train.csv')

X_pretest = pd.read_csv("X_pretest_processed.csv")

y_pretest = pd.Series.from_csv('y_pretest.csv')
# -

X_train = X_train.drop('Unnamed: 0', axis=1)
X_pretest = X_pretest.drop('Unnamed: 0', axis=1)

X_pretest.shape

X_train.shape

# +
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
forest_clf.fit(X_train, y_train)

y_pretest_pred = forest_clf.predict(X_pretest)

score = accuracy_score(y_pretest_pred, y_pretest)

print('accuracy: {}'.format(score))
# -

# 72% percent accuracy is pretty good! but remember, if the model predicts that all of the entries pay off their loans, we would get a 78.77% accuracy.

print('Pays off loans:', round(y_train.value_counts()[1]/len(y_train) * 100,2), '% of the dataset')
print('Does not pay off loans:', round(y_train.value_counts()[0]/len(y_train) * 100,2), '% of the dataset')

# # let's do this with cross validation

# +
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    


# +
# %%capture

forest_scores = cross_val_score(forest_clf, X_train, y_train,
                                scoring="accuracy", cv=10)


# -

display_scores(forest_scores)

# # damn, the RMSE error is pretty bad
#
# # let's try GridSearch to find optimal setting for random forest

# +
from sklearn.model_selection import GridSearchCV

display_scores(forest_rmse_scores)
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_clf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(forest_clf, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)


grid_search.fit(X_train, y_train)
# -

grid_search.best_params_

grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

best_model = grid_search.best_estimator_

rmse_value(X_pretest, y_pretest, best_model)

# # let's use confusion matrix to evaluate the error properly 

# +
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_pretest_pred = cross_val_predict(best_model, X_pretest, y_pretest, cv=3)

y_pretest_pred = y_pretest_pred.round()

confusion_mat = confusion_matrix(y_pretest, y_pretest_pred)

sns.set_style("white")
plt.matshow(confusion_mat, cmap=plt.cm.gray)
plt.show()

# +
row_sums = confusion_mat.sum(axis=1, keepdims=True)
normalised_confusion_mat = confusion_mat/row_sums

plt.matshow(normalised_confusion_mat, cmap=plt.cm.gray)
print(confusion_mat, "\n")
print(normalised_confusion_mat)
# -

# # basically predicted most Loan Status to be 1 

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pretest_pred, y_pretest))


# +
from sklearn.metrics import precision_score, recall_score, f1_score

print('the precision score is : ', precision_score(y_pretest, y_pretest_pred))
print('the recall score is : ', recall_score(y_pretest, y_pretest_pred))
print('the f1 score is : ', f1_score(y_pretest, y_pretest_pred))
# -

y_train.value_counts()

y_pretest.shape

# # let's look at the precision and recall curve

# +
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_pretest_pred, y_pretest)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds): 
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall") 
    plt.xlabel("Threshold")
    plt.legend(loc="lower left") 
    plt.ylim([0, 1])
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds) 
plt.show()
# -

# # let's look at the learning curve

# +
from sklearn.model_selection import train_test_split

def plot_learning_curves(X, y, model, title, starting_point):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(starting_point, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train") 
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel('Training set size')
    plt.ylabel('RMSE')

    plt.show()


# -

plot_learning_curves(X_train, y_train, best_model, "Best Random Forest predictor", 1)


