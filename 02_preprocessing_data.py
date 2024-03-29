# -*- coding: utf-8 -*-
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
pd.set_option('display.max_rows', 30)

df = pd.read_csv("credit_train.csv")


# -

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


missing_values_table(df)

# +
df = df.drop('Loan ID', axis =1)
df = df.drop('Customer ID', axis=1)

#remember that "Months since last delinquent" had more than 50% missing, so let's drop this column too

df = df.drop('Months since last delinquent', axis=1)
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

# here we make a X_pretest set that has 2/10 of the rows that don't have nan values. that is 2/10 * 77271 = 15450, 
# so 15% of the training dataset

X_pretest, remaining_nonan = train_test_split(nonan_df, train_size= 0.2, random_state = 42)
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

df.shape

# # Okay, let's fill in the nans for Credit Score, Annual Income, Years in current job, Bankruptcies and Tax Liens

missing_values_table(df)


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
fill_mode(df, ['Years in current job', 'Tax Liens'])
fill_mean(df, ['Annual Income', 'Credit Score'])
fill_median(df, ['Maximum Open Credit'])

df['Bankruptcies'].fillna(1.0, inplace=True)


# -

def print_mmm(df, attrib):
    print('the mode: {}',format(df[attrib].mode()))
    print('the median: {}',format(df[attrib].median()))
    print('the mean: {}',format(df[attrib].mean()))
    print('the max: {}',format(df[attrib].max()))
    print('the min: {}',format(df[attrib].min()))


missing_values_table(df)

# # Great, filled all the nan values! - now let's remove the possible outliers
#
# # First let's see what the box plots look like first

# +
f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Loan Status", y="Monthly Debt", data=df, ax=axes[0])
axes[0].set_title('Monthly Debt')

sns.boxplot(x="Loan Status", y="Annual Income", data=df, ax=axes[1])
axes[1].set_title('Annual Income')

sns.boxplot(x="Loan Status", y="Current Credit Balance", data=df, ax=axes[2])
axes[2].set_title('Current Credit Balance')

sns.boxplot(x="Loan Status", y="Credit Score", data=df, ax=axes[3])
axes[3].set_title('Credit Score')

plt.show()


# -

# So many outliers!
#
# # here's a simple box plotting function

# +
def simple_box_plot(df, x_attrib, y_attrib):
    f, axes = plt.subplots(ncols=1, figsize=(7,7))

    sns.boxplot(x=y_attrib, y=x_attrib, data=df)
    axes.set_title(x_attrib)

    
    for i in df[y_attrib].unique():
        print("Median for '{}': {}".format(i, df[x_attrib][df[y_attrib] == i].median()))

plt.show()
    
# -

def remove_outliers(df, x_attrib, y_attrib):

    for i in df[y_attrib].unique():
        
        m, n = df.shape
        print('Number of rows: {}'.format(m))
        
        remove_list = df[x_attrib].loc[df[y_attrib] == i].values
        q25, q75 = np.percentile(remove_list, 25), np.percentile(remove_list, 75)
        print('Lower Quartile: {} | Upper Quartile: {}'.format(q25, q75))
        iqr = q75 - q25
        print('iqr: {}'.format(iqr))

        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        print('Cut Off: {}'.format(cut_off))
        print('Lower Extreme: {}'.format(lower))
        print('Upper Extreme: {}'.format(upper))

        outliers = [x for x in remove_list if x < lower or x > upper]
        print('Number of Outliers for {} Cases: {}'.format(i, len(outliers)))
        print('outliers:{}'.format(outliers))

        for d in outliers:
            #delete_row = new_df[new_df[y_attrib]==i].index
            #new_df = new_df.drop(delete_row)
            df = df[df[x_attrib] != d]
        
        m, n = df.shape
        print('Number of rows for new dataframe: {}\n'.format(m))
    
    new_df = df
    
    print('----' * 27)
    return new_df


df.head()

# # Monthly Debt

simple_box_plot(df, 'Monthly Debt', 'Loan Status')

new_df = remove_outliers(df, 'Monthly Debt', 'Loan Status')

simple_box_plot(new_df, 'Monthly Debt', 'Loan Status')

# # Annual Income

simple_box_plot(new_df, 'Annual Income', 'Loan Status')

new_df = remove_outliers(new_df, 'Annual Income', 'Loan Status')

simple_box_plot(new_df, 'Annual Income', 'Loan Status')

# # Current Credit Balance

simple_box_plot(new_df, 'Current Credit Balance', 'Loan Status')

new_df = remove_outliers(new_df, 'Current Credit Balance', 'Loan Status')

simple_box_plot(new_df, 'Current Credit Balance', 'Loan Status')

new_df.head()

# # Credit Score

simple_box_plot(new_df, 'Credit Score', 'Loan Status')

new_new_df = remove_outliers(new_df, 'Credit Score', 'Loan Status')

simple_box_plot(new_new_df, 'Credit Score', 'Loan Status')

new_df = new_new_df

# # Current Loan Amount

simple_box_plot(new_df, 'Current Loan Amount', 'Loan Status')

new_new_df = remove_outliers(new_df, 'Current Loan Amount', 'Loan Status')

simple_box_plot(new_new_df, 'Current Loan Amount', 'Loan Status')

new_df = new_new_df

# # Maximum Open Credit
#

simple_box_plot(new_df, 'Maximum Open Credit', 'Loan Status')

new_new_df = remove_outliers(new_df, 'Maximum Open Credit', 'Loan Status')

simple_box_plot(new_new_df, 'Maximum Open Credit', 'Loan Status')

new_df = new_new_df

new_df.shape


# # Let's look at some of the categorical variables and see what needs to be changed or removed

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


# # Purpose 

plot_bar_graphs(new_df, 'Purpose', 'Loan Status')

# other and Other are the same, let's correct this. Let's classify moving and Home Improvements as the same thing, etc.

new_df['Purpose'].replace({ 'other':'Other', 'renewable_energy': 'Other',
                        'Buy a Car':'major_purchase',
                        'moving':'Home Improvements',
                        'Take a trip': 'vacation',
                        'Educational Expenses': 'Other',
                        'vacation':'Take a Trip',
                        'wedding': 'major_purchase'},inplace=True)

new_df['Purpose'].value_counts()   

plot_bar_graphs(new_df, 'Purpose', 'Loan Status')

# need to do the same thing here with X_pretest

X_pretest['Purpose'].value_counts()

X_pretest['Purpose'].replace({ 'other':'Other', 'renewable_energy': 'Other',
                        'Buy a Car':'major_purchase',
                        'moving':'Home Improvements',
                        'Take a trip': 'vacation',
                        'Educational Expenses': 'Other',
                        'vacation':'Take a Trip',
                        'wedding': 'major_purchase'},inplace=True)

X_pretest['Purpose'].value_counts()

# # Number of Credit Problems

plot_bar_graphs(new_df, 'Number of Credit Problems', 'Loan Status')

# this doesn't look right, if you have 7, 8, 10, 11 or 15 credit problems, you will be predicted to have fully paid everything... maybe it's worth just removing these

# +
new_df = new_df[new_df['Number of Credit Problems'] != 8.0]
new_df = new_df[new_df['Number of Credit Problems'] != 10.0]
new_df = new_df[new_df['Number of Credit Problems'] != 11.0]
new_df = new_df[new_df['Number of Credit Problems'] != 15.0]

new_df['Number of Credit Problems'].value_counts()
# -

plot_bar_graphs(new_df, 'Number of Credit Problems', 'Loan Status')

# # Bankruptcies

plot_bar_graphs(new_df, 'Bankruptcies', 'Loan Status')

# let's remove the row that has Bankruptcies = 6.0 

new_df = new_df[new_df['Bankruptcies'] != 6.0]

plot_bar_graphs(new_df, 'Bankruptcies', 'Loan Status')

# # Tax Liens

plot_bar_graphs(new_df, 'Tax Liens', 'Loan Status')

# this looks ok to me

# # Home Ownership

plot_bar_graphs(new_df, 'Home Ownership', 'Loan Status')

# +
new_df['Home Ownership'].replace({ 'HaveMortgage':'Home Mortgage'},inplace=True)

X_pretest['Home Ownership'].replace({ 'HaveMortgage':'Home Mortgage'},inplace=True)


# -

# # Make extra features/columns? ... let's try this another time

new_df.head()

# # Change the y variables into integers - here I've made Charged Off = 1 and Fully Paid = 0

# +
# change the varialbes Fully Paid and Charged Off into 0 and 1

new_df['Loan Status'] = (new_df['Loan Status'] == 'Charged Off').astype(int)
X_pretest['Loan Status'] = (X_pretest['Loan Status'] == 'Charged Off').astype(int)
# -

new_df['Loan Status'].value_counts()

# # Let's have a quick look at the correlation matrix for this
#

# +
import seaborn as sns

def corr_matrix(df, attribute_list, key_attribute):
    new_df = pd.DataFrame()
    for i in attribute_list:
        new_df[i] = df[i]
            
    matrix = new_df.corr()
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, vmax=.8, square=True, cmap="YlGnBu")
    
    print(matrix[key_attribute].sort_values(ascending=False))
    
    return 


# -

list_attribs = ['Loan Status', "Current Loan Amount", "Annual Income", "Monthly Debt",
                 'Years of Credit History',
                 'Number of Open Accounts', 'Number of Credit Problems',
                 'Current Credit Balance', 'Maximum Open Credit',
                 'Bankruptcies', 'Tax Liens']

corr_matrix(new_df, list_attribs, 'Loan Status')





# # Let's export this data before we put it through the pipeline

# +
y_train = new_df['Loan Status']
y_pretest = X_pretest['Loan Status']

X_train = new_df.drop(['Loan Status'], axis=1)
X_pretest = X_pretest.drop(['Loan Status'], axis=1)
# -

y_train.value_counts()

X_train.shape

X_pretest.shape

X_train.to_csv(r'X_train.csv')
X_pretest.to_csv(r'X_pretest.csv')
y_train.to_csv(r'y_train.csv')
y_pretest.to_csv(r'y_pretest.csv')

# # Now let's use pipelines to scale the numerical data and one hot encode the categorical data

# +
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class selector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values
      
num_attributes = ["Current Loan Amount", "Annual Income", "Monthly Debt",
                 'Years of Credit History',
                 'Number of Open Accounts', 'Number of Credit Problems',
                 'Current Credit Balance', 'Maximum Open Credit',
                 'Bankruptcies', 'Tax Liens']

num_pipeline = Pipeline([
            ('selector', selector(num_attributes)),
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
                    ])

cat_attributes = ['Term', 'Years in current job', 'Home Ownership', 'Purpose']

cat_pipeline = Pipeline([
                ('selector', selector(cat_attributes)),
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('cat_encoder', OneHotEncoder(sparse=False)),
])

# +
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

# +
X_train_processed = full_pipeline.fit_transform(X_train)
X_train_processed = pd.DataFrame(X_train_processed)


# note that we are using transform() on X_pretest 
#we standardize our training dataset, we need to keep the parameters (mean and standard deviation for each feature). 
#Then, we use these parameters to transform our test data and any future data later on
#fit() just calculates the parameters (e.g. 𝜇 and 𝜎 in case of StandardScaler) 
#and saves them as an internal objects state. 
#Afterwards, you can call its transform() method to apply the transformation to a particular set of examples.
X_pretest_processed = full_pipeline.transform(X_pretest)
X_pretest_processed = pd.DataFrame(X_pretest_processed)



# -
X_train_processed.shape


X_pretest_processed.shape


X_train_processed.to_csv(r'X_train_processed.csv')
X_pretest_processed.to_csv(r'X_pretest_processed.csv')










