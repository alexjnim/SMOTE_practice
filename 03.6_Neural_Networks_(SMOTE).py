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

y_train.value_counts()

# # First let's try neural networks without any changes first 

# +
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

n_inputs = X_train.shape[1]

undersample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
# -

undersample_model.summary()

undersample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

undersample_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True, verbose=2)

undersample_predictions = undersample_model.predict(X_pretest, batch_size=200, verbose=0)

undersample_fraud_predictions = undersample_model.predict_classes(X_pretest, batch_size=200, verbose=0)

undersample_fraud_predictions.shape

# +
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score

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

confusion_matrices(y_pretest, undersample_fraud_predictions)

# # terrible! 

# +
n_inputs = Xsm_train.shape[1]

oversample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

# +
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

smt = SMOTE()

def KFold_SMOTE_model_scores(X_df, y):
    
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
        
        #here we build the the neural network 
        n_inputs = X_train_SMOTE.shape[1]
        model = Sequential([Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
                            Dense(32, activation='relu'),
                            Dense(2, activation='softmax')
                            ])
        model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        print("---" * 7)
        print("\n")
        
        model.fit(X_train_SMOTE, y_train_SMOTE, validation_split=0.2,
                  batch_size=300, epochs=20, shuffle=True, verbose=2)
        
        #find the accuracy score of the validation set
        y_val_pred = model.predict_classes(X_val, batch_size=200, verbose=0)
        scores.append(recall_score(y_val, y_val_pred))
        
        #find the best model based on the accuracy score
        if recall_score(y_val, y_val_pred) == max(scores):
            best_model = model
    
    return scores, best_model
# -

scores, best_model = KFold_SMOTE_model_scores(X_train, y_train)

# +
y_train_pred = best_model.predict_classes(X_train)

confusion_matrices(y_train, y_train_pred)

# +
y_pretest_pred = best_model.predict_classes(X_pretest)

confusion_matrices(y_pretest, y_pretest_pred)
# -

# # not very good at all to be honest 


