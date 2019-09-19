# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
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

y_train = pd.read_csv('y_train.csv', header = None, index_col = 0, squeeze = bool)

X_pretest = pd.read_csv("X_pretest_processed.csv")

y_pretest = pd.read_csv('y_pretest.csv', header = None, index_col = 0, squeeze = bool)
# -

X_train = X_train.drop('Unnamed: 0', axis=1)
X_pretest = X_pretest.drop('Unnamed: 0', axis=1)

X_pretest.shape

X_train.shape

y_train.value_counts()

# # Let's try simply RandomForest first, no cross validation

# +
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
forest_clf.fit(X_train, y_train)

y_train_pred = forest_clf.predict(X_train)
#y_pretest_pred = forest_clf.predict(X_pretest)

score = accuracy_score(y_train, y_train_pred)
recall_score = recall_score(y_train, y_train_pred)
precision_score = precision_score(y_train, y_train_pred)

print('accuracy: {}'.format(score))
print('recall: {}'.format(recall_score))
print('precision: {}'.format(precision_score))
# -

# # 97% accuracy?? surely not, it must have overfitted. let's check with cross val

print('Pays off loans:', round(y_train.value_counts()[0]/len(y_train) * 100,2), '% of the dataset')
print('Does not pay off loans:', round(y_train.value_counts()[1]/len(y_train) * 100,2), '% of the dataset')

# +
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

def confusion_matrices(y, y_pred):
    y_pred = y_pred.round()
        
    confusion_mat = confusion_matrix(y, y_pred)

    sns.set_style("white")
    plt.matshow(confusion_mat, cmap=plt.cm.gray)
    plt.show()

    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    normalised_confusion_mat = confusion_mat/row_sums
    
    print(confusion_mat, "\n")
    print(normalised_confusion_mat)
    
    plt.matshow(normalised_confusion_mat, cmap=plt.cm.gray)
    plt.show()

    print('the precision score is : ', precision_score(y, y_pred))
    print('the recall score is : ', recall_score(y, y_pred))
    print('the f1 score is : ', f1_score(y, y_pred))
    print('the accuracy score is : ', accuracy_score(y, y_pred))
    
    return


# -

confusion_matrices(y_train, y_train_pred)

# # probably overfitted, just memorised the answers to each data point! let's confirm this with pretest set

# +
y_pretest_pred = forest_clf.predict(X_pretest)

confusion_matrices(y_pretest, y_pretest_pred)
# -

# # as I thought, it simply predicted that all points are 0 = fully paid

# # let's further confrim this with Cross Validation to check this against the validation set

# +
from sklearn.model_selection import cross_val_score

def display_scores(scores, name):
    print("{} Scores: {}".format(name,scores))
    print("{} Mean: {}".format(name,scores.mean()))
    print("{} Standard deviation: {}".format(name,scores.std()))
    print("\n")
    


# +
score = ['accuracy', 'precision', 'recall']


for i in score:
    forest_scores = cross_val_score(forest_clf, X_train, y_train,
                                scoring=i, cv=10)
    display_scores(forest_scores, i)
# -

# # 74%? This is more like it, but I suspect that it has simply predicted everything to be 0 = Fully Paid, evidently seen by a recall of 0.08 

# +
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(forest_clf, X_train, y_train, cv=10)
# -

confusion_matrices(y_train, y_train_pred)

# # exactly as I suspected! 

# # let's try GridSearch just to find the optimal hyper parameters, probably won't make the results any better though. let's score this based on 'recall' and select the best model based on the recall score

# +
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_clf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(forest_clf, param_grid, cv=5,
                           scoring='recall', return_train_score=True)


grid_search.fit(X_train, y_train)
# -

grid_search.best_params_

grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)

best_model = grid_search.best_estimator_

feature_importances = grid_search.best_estimator_.feature_importances_ 
feature_importances

# # here we can decide which features are important or not, and therefore drop them! but let's do this later

X_train.columns

# # let's use confusion matrix on training set to see what these accuracy scores are coming from

y_train_pred = best_model.predict(X_train)

confusion_matrices(y_train, y_train_pred)

# # let's use confusion matrix on pretest set to evaluate the error properly 

# +
#y_pretest_pred = best_model.predict(X_pretest)
y_pretest_pred = cross_val_predict(best_model, X_pretest, y_pretest, cv=3)

confusion_matrices(y_pretest, y_pretest_pred)
# -

# # basically predicted most Loan Status to be 0 (Fully Paid). This is terrible 

# # these scores are terrible! - let's try this with SMOTE

# +
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

smt = SMOTE()

def KFold_SMOTE_model_scores(X_df, y, model):
    
    scores = []
    cv = KFold(n_splits=5, random_state=42, shuffle=False)
    
    # need to reset the indices as the 
    X_df = X_df.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    #this will shuffle through 5 different training and validation data splits 
    for train_index, val_index in cv.split(X_df):
        
        X_train = X_df.loc[train_index]
        y_train = y.loc[train_index]
        
        X_val = X_df.loc[val_index]
        y_val = y.loc[val_index]   
        
        print('Before OverSampling, the shape of X_train: {}'.format(X_train.shape))
        print('Before OverSampling, the shape of y_train: {} \n'.format(y_train.shape))

        print("Before OverSampling, counts of label 'Y': {}".format(sum(y_train==1)))
        print("Before OverSampling, counts of label 'N': {} \n".format(sum(y_train==0)))
        
        
        # this will create minority class data points such that y_train has 50% == 1 and 50% == 0
        X_train_SMOTE, y_train_SMOTE = smt.fit_sample(X_train, y_train)
        
        print('After OverSampling, the shape of X_train: {}'.format(X_train_SMOTE.shape))
        print('After OverSampling, the shape of y_train: {} \n'.format(y_train_SMOTE.shape))

        print("After OverSampling, counts of label 'Y': {}".format(sum(y_train_SMOTE==1)))
        print("After OverSampling, counts of label 'N': {} \n".format(sum(y_train_SMOTE==0)))
        
        print("---" * 7)
        print("\n")
        
        model.fit(X_train_SMOTE, y_train_SMOTE)
        
        #find the accuracy score of the validation set
        y_val_pred = model.predict(X_val)
        scores.append(recall_score(y_val, y_val_pred))
        
        #find the best model based on the accuracy score
        if recall_score(y_val, y_val_pred) == max(scores):
            best_model = model
    
    return scores, best_model

# +
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=30, random_state=42)

scores, best_model = KFold_SMOTE_model_scores(X_train, y_train, forest_clf)
# -

scores = np.array(scores)
display_scores(scores, 'recall')

# # the mean recall on validation set during cross val is 0.2. this isn't promising for pretest data
#
# # this time, 0 and 1 are 50/50 from SMOTE. so predicting that everything is 0 or 1 will give an accuracy of 50. 

# # let's check this quickly on training data -> and then look at this for test data

# +
y_train_pred = best_model.predict(X_train)

confusion_matrices(y_train, y_train_pred)
# -

# # it does really well on training data! it has most likely overfitted. Let's look at pretest set 

# +
y_pretest_pred = best_model.predict(X_pretest)

confusion_matrices(y_pretest, y_pretest_pred)
# -

# # better with SMOTE, but still not that great! it's just predicting that y = 0 (fully paid) for everything


