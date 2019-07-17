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

np.random.seed(42)

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

df.head()

# let's drop Loan ID and Customer ID as this won't be used for modelling

df.shape

df = df.drop('Loan ID', axis =1)
df = df.drop('Customer ID', axis=1)

df.isnull().any(axis=0)

df.isnull().sum()

df = df.dropna(subset=['Loan Status']) 

df.isnull().sum()

df.shape

df['Loan Status'].value_counts()


# # bar plot function 

def plot_bar_graphs(df, attribute, y):
    plt.figure(1)
    plt.subplot(131)
    df[attribute].value_counts(normalize=True).plot.bar(figsize=(22,4),title= attribute)
    
    crosstab = pd.crosstab(df[attribute], df[y])
    crosstab.div(crosstab.sum(1).astype(float), axis=0).plot.bar(stacked=True)
    crosstab.plot.bar(stacked=True)
    
    res = df.groupby([attribute, y]).size().unstack()
    tot_col = 0
    for i in range(len(df[y].unique())):
        tot_col = tot_col + res[res.columns[i]] 
        
    for i in range(len(df[y].unique())):    
        res[i] = (res[res.columns[i]]/tot_col)
    
    res = res.sort_values(by = [0], ascending = True)
    print(res)
    
    return


# # Term

df['Term'].value_counts()

plot_bar_graphs(df, 'Term', 'Loan Status')

# looks like people who have long term loans are more likely to default

# # Home Ownership

df['Home Ownership'].value_counts()

plot_bar_graphs(df, 'Home Ownership', 'Loan Status')

# people that are renting are more likely to be default. surprisingly, you'd think that people who own their own homes are less likely to defualt as they are more financially stable?

# # Purpose 

df['Purpose'].value_counts()

plot_bar_graphs(df, 'Purpose', 'Loan Status')

# makes sense that people taking about business loans are making a risk, particularly small businesses 

# # Histogram function

# +
dodger_blue = '#1E90FF'
crimson = '#DC143C'
lime_green = '#32CD32'
red_wine = '#722f37'
white_wine = '#dbdd46' 

def plot_histograms(df, x_attribute, n_bins, x_max, y_attribute):
    
    #this removes the rows with nan values for this attribute  
    df = df.dropna(subset=[x_attribute]) 
    
    print ("Mean: {:0.2f}".format(df[x_attribute].mean()))
    print ("Median: {:0.2f}".format(df[x_attribute].median()))
           
    df[x_attribute].hist(bins= n_bins, color= crimson)
    
    #this plots the mean and median 
    plt.plot([df[x_attribute].mean(), df[x_attribute].mean()], [0, 60000],
        color='black', linestyle='-', linewidth=2, label='mean')
    plt.plot([df[x_attribute].median(), df[x_attribute].median()], [0, 60000],
        color='black', linestyle='--', linewidth=2, label='median')
    
    plt.xlim(xmin=0, xmax = x_max)
    plt.xlabel(x_attribute)
    plt.ylabel('COUNT')
    plt.title(x_attribute)
    plt.legend(loc='best')
    plt.show()

    df[df[ y_attribute]==0][x_attribute].hist(bins=n_bins, color = crimson, label='No default')

    print ("Y Mean: {:0.2f}".format(df[df[y_attribute]==0][x_attribute].mean()))
    print ("Y Median: {:0.2f}".format(df[df[ y_attribute]==0][x_attribute].median()))
    
    plt.plot([df[df[ y_attribute]==0][x_attribute].mean(), df[df[ y_attribute]==0][x_attribute].mean()], 
            [0, 60000], color='r', linestyle='-', linewidth=2, label='Y mean') 
    plt.plot([df[df[ y_attribute]==1][x_attribute].mean(), df[df[ y_attribute]==1][x_attribute].mean()], 
            [0, 60000], color='b', linestyle='-', linewidth=2, label='N mean')
 
    df[df[ y_attribute]==1][x_attribute].hist(bins=n_bins, color = lime_green, label='Default')
    
    print ("N Mean: {:0.2f}".format(df[df[ y_attribute]==1][x_attribute].mean()))
    print ("N Median: {:0.2f}".format(df[df[ y_attribute]==1][x_attribute].median()))
    
    plt.plot([df[df[ y_attribute]==0][x_attribute].median(), df[df[ y_attribute]==0][x_attribute].median()], 
            [0, 60000], color='r', linestyle='--', linewidth=2, label='Y median') 
    plt.plot([df[df[ y_attribute]==1][x_attribute].median(), df[df[ y_attribute]==1][x_attribute].median()], 
            [0, 60000], color='b', linestyle='--', linewidth=2, label='N median')
    
    plt.xlim(xmin=0, xmax = x_max)
    
    plt.title(x_attribute)
    plt.xlabel(x_attribute)
    plt.ylabel('COUNT')
    plt.legend(loc='best')
    plt.show()    
    return
    


# -


