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

# +
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

y_train_pred = sgd_clf.predict(X_train)

score = accuracy_score(y_train, y_train_pred)

print('accuracy: {}'.format(score))
# -

print('Pays off loans:', round(y_train.value_counts()[0]/len(y_train) * 100,2), '% of the dataset')
print('Does not pay off loans:', round(y_train.value_counts()[1]/len(y_train) * 100,2), '% of the dataset')

# # this looks like the SGDClassifier has just predicted that everything is 0 = pays off loans

# # let's do this with Cross Validation to check the actual accuracy as it can test this against the validation set

# +
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    


# +
# %%capture

sgd_scores = cross_val_score(sgd_clf, X_train, y_train,
                                scoring="accuracy", cv=10)



# -

display_scores(sgd_scores)

# # most likely predicted everything to be 0 = pays off loans

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


# +
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=10)
# -

confusion_matrices(y_train, y_train_pred)

# # this is terrible! - will SMOTE help?

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
        scores.append(accuracy_score(y_val, y_val_pred))
        
        #find the best model based on the accuracy score
        if accuracy_score(y_val, y_val_pred) == max(scores):
            best_model = model
    
    return scores, best_model
# -

scores, best_model = KFold_SMOTE_model_scores(X_train, y_train, sgd_clf)

# +
y_train_pred = best_model.predict(X_train)

confusion_matrices(y_train, y_train_pred)
# -

# # this is significantly better! much better recall.
#
#
# # let's try this on the the pretest data

# +
y_pretest_pred = best_model.predict(X_pretest)

confusion_matrices(y_pretest, y_pretest_pred)
# -

# # this isn't great, but it is better than the Random Forest model. So SMOTE can be really helpful! 


