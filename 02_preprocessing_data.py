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

df = df.dropna(subset=['Loan Status']) 

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


# -

df.head()

fill_mode(X_train, ['Credit Score', 'Annual Income', 'Years in curent job'])
fill_mean(X_train, ['Months since last delinquent'])


def print_mmm(df, attrib):
    print('the mode: {}',format(df[attrib].mode()))
    print('the median: {}',format(df[attrib].median()))
    print('the mean: {}',format(df[attrib].mean()))
    print('the max: {}',format(df[attrib].max()))
    print('the min: {}',format(df[attrib].min()))


print_mmm(df, 'Months since last delinquent')


