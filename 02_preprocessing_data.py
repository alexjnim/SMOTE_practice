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

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# +
import pandas as pd
pd.set_option('display.max_rows', 30)

df = pd.read_csv("credit_train.csv")
# -

df.isnull().sum()

# # there seems to be 514 rows entirely filled with nans. let's remove the ones based on Loan Status at least

df = df.dropna(subset=['Loan Status']) 

df.isnull().sum()

# # let's split this off and make a pre-test set, where none of the rows have nan values, so that when we test our model later, the pre-test data won't have any  artificially feature values 

# +
# seperate the dataframe into one with all rows that have nan values, and the other with no nan values

nan_df = df[df.isna().any(axis=1)]
nonan_df = df.dropna()
# -

nan_df.shape

nan_df.isnull().sum()

nonan_df.shape

nonan_df.isnull().sum()

# +
from sklearn.model_selection import train_test_split

# here we make a X_pretest set that has 4/10 of the rows that have nan

X_pretest, remaining_nonan = train_test_split(nonan_df, train_size= 0.4, random_state = 42)
# -

X_pretest.shape

X_pretest.isnull().any(axis=0)

df = pd.concat([nan_df, remaining_nonan])


def print_totshape(df1, df2):
    m,n = df1.shape
    a,b = df2.shape
    
    print(m+a)
    return


print_totshape(df, X_pretest)

df.isnull().sum()


# # Okay, let's fill in the nans for Credit Score, Annual Income, Years in current job, Months since last delinquent, Bankruptcies and Tax Liens

# +
def fill_mode(df, attribute_list):
    for i in attribute_list:
        print(i)
        df[i].fillna(df[i].mode()[0], inplace=True)
    return df

def fill_mean(df, attribute_list):
    for i in attribute_list:
        print(i)
        df[i].fillna(df[i].mean(), inplace=True)
    return df

def fill_median(df, attribute_list):
    for i in attribute_list:
        print(i)
        df[i].fillna(df[i].median(), inplace=True)
    return df


# -

df.head()

# +
fill_mode(df, ['Credit Score', 'Annual Income', 'Years in current job', 'Tax Liens'])
fill_mean(df, ['Months since last delinquent'])
fill_median(df, ['Maximum Open Credit'])

df['Bankruptcies'].fillna(1.0, inplace=True)


# -

def print_mmm(df, attrib):
    print('the mode: {}',format(df[attrib].mode()))
    print('the median: {}',format(df[attrib].median()))
    print('the mean: {}',format(df[attrib].mean()))
    print('the max: {}',format(df[attrib].max()))
    print('the min: {}',format(df[attrib].min()))


df.isnull().sum()

# # Great, filled all the nan values! - now let's remove the possible outliers

df.shape


