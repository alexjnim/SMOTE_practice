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

# # Number of Credit Problems 

df['Number of Credit Problems'].value_counts()

plot_bar_graphs(df, 'Number of Credit Problems', 'Loan Status')

# makes sense, when you start to have more than 5 credit problems, your likelihood of defaulting is a lot higher

# # Years in current job

df['Years in current job'].value_counts()

plot_bar_graphs(df, 'Years in current job', 'Loan Status')

# looks like the number of years that you've been in your current job does not change how likely you are to default

# # Bankruptcies

df['Bankruptcies'].value_counts()

plot_bar_graphs(df, 'Bankruptcies', 'Loan Status')

# # tax liens

# little influence 

df['Tax Liens'].value_counts()

plot_bar_graphs(df, 'Tax Liens', 'Loan Status')

# as you start to get more than 3 tax liens, you're more likely to defaul it seems

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
    plt.plot([df[x_attribute].mean(), df[x_attribute].mean()], [0, 10000],
        color='black', linestyle='-', linewidth=2, label='mean')
    plt.plot([df[x_attribute].median(), df[x_attribute].median()], [0, 10000],
        color='black', linestyle='--', linewidth=2, label='median')
    
    plt.xlim(xmin=0, xmax = x_max)
    plt.xlabel(x_attribute)
    plt.ylabel('COUNT')
    plt.title(x_attribute)
    plt.legend(loc='best')
    plt.show()

    df[df[ y_attribute]=='Fully Paid'][x_attribute].hist(bins=n_bins, color = crimson, label='No default')

    print ("Fully Paid Mean: {:0.2f}".format(df[df[y_attribute]=='Fully Paid'][x_attribute].mean()))
    print ("Fully Paid Median: {:0.2f}".format(df[df[ y_attribute]=='Fully Paid'][x_attribute].median()))
    
    plt.plot([df[df[ y_attribute]=='Fully Paid'][x_attribute].mean(), df[df[ y_attribute]=='Fully Paid'][x_attribute].mean()], 
            [0, 5000], color='r', linestyle='-', linewidth=2, label='Y mean') 
    plt.plot([df[df[ y_attribute]=='Charged Off'][x_attribute].mean(), df[df[ y_attribute]=='Charged Off'][x_attribute].mean()], 
            [0, 5000], color='b', linestyle='-', linewidth=2, label='N mean')
 
    df[df[ y_attribute]=='Charged Off'][x_attribute].hist(bins=n_bins, color = lime_green, label='Default')
    
    print ("Charged Off Mean: {:0.2f}".format(df[df[ y_attribute]=='Charged Off'][x_attribute].mean()))
    print ("Charged Off Median: {:0.2f}".format(df[df[ y_attribute]=='Charged Off'][x_attribute].median()))
    
    plt.plot([df[df[ y_attribute]=='Fully Paid'][x_attribute].median(), df[df[ y_attribute]=='Fully Paid'][x_attribute].median()], 
            [0, 5000], color='r', linestyle='--', linewidth=2, label='Y median') 
    plt.plot([df[df[ y_attribute]=='Charged Off'][x_attribute].median(), df[df[ y_attribute]=='Charged Off'][x_attribute].median()], 
            [0, 5000], color='b', linestyle='--', linewidth=2, label='N median')
    
    plt.xlim(xmin=0, xmax = x_max)
    
    plt.title(x_attribute)
    plt.xlabel(x_attribute)
    plt.ylabel('COUNT')
    plt.legend(loc='best')
    plt.show()    
    return
    


# -

df.head()

# # Monthly Debt

plot_histograms(df, 'Monthly Debt', 150, 80000, 'Loan Status')

# the means and medians are roughly around the same region, but those for Charged Off slightly higher. It looks like your monthly debt does not affect this too much.

# # Annual Income 

df['Annual Income'].max()

plot_histograms(df, 'Annual Income', 1000, 7000000, 'Loan Status')

# looks like people with a smaller annual income are more likely to default according to the mean and medians. this makes sense

# # Number of Open Accounts

plot_histograms(df, 'Number of Open Accounts', 51, 100, 'Loan Status')

# # Current Loan Amount

plot_histograms(df, 'Current Loan Amount', 10000, 1000000, 'Loan Status')

# # Current Credit Balance

plot_histograms(df, 'Current Credit Balance', 2000, 1000000, 'Loan Status')

# # Maximum Open Credit

plot_histograms(df, 'Maximum Open Credit', 4000, 1000000, 'Loan Status')

# # Credit Score

plot_histograms(df, 'Credit Score', 500, 2000, 'Loan Status')


