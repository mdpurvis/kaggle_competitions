# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:06:43 2020

@author: mdpur
"""

import pandas as pd
import numpy as np
import re
from scipy.stats import ttest_ind
import scipy.stats
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] =(30,12)

#changes
# define module to create frequency distribution of variables
def freq_dist(df, col):
    df_fd = pd.DataFrame(data=df[col].value_counts())
    df_fd['pct_total'] = df_fd[col] / df_fd[col].sum()
    print(df_fd.head(15))
    
    
# define module to convert individual values in to buckets, creating corresponding labels; 
# outputs frequency distribution module above
def creat_bins(df, new_col, col, bins=[], labels=[]):
    df[new_col] = pd.cut(
    x=df[col], 
    bins=bins, 
    labels=labels)
    freq_dist(df, new_col)
    
    
#Module used to take a field in dollars seen as an 'object' and convert in to an Integer    
def convert_dollars(cols):
    df[cols] = df[cols].str.replace('#NULL!', '0.0')
    df[cols] = df[cols].str.replace(',', '')
    df[cols] = df[cols].str.replace('$', '')
    df[cols] = df[cols].str.replace(' ', '')
    df[cols] = df[cols].str.replace(')', '')
    df[cols] = df[cols].str.replace('-', '0.0')
    df[cols] = df[cols].str.replace('(', '-')
    df[cols] = df[cols].astype(float)
    df[cols] = df[cols].replace('.0', '')
    df[cols] = df[cols].astype(int)
    print('Converting', cols, 'Dollars to Numbers Complete')    
    
    
#Creates Histogram for Comparitive Purposes
def review_detail(df, equil, col, num, col2, num2, col3, meas, sub, type):    
    if type == 'total':
        df_last_month = df.loc[(df[col] != num) & (df[col2] != num2)]
        df_last_month = df_last_month.groupby(by=[col, col3, col2]).agg({'Total'+equil:meas})
        df_last_month.rename(columns = {'index':col3}, inplace=True)
        df_last_month['pct_total'] = ((df_last_month['Total'+equil] / df_last_month['Total'+equil].sum())*100).astype(int)
        df_last_month = pd.DataFrame(df_last_month)
        df_last_month = df_last_month.T
        print(df_last_month)
        df_last_month.drop_duplicates(inplace=True)
        df_last_month = df_last_month.T
        df_last_month.drop(columns=['pct_total'], inplace=True)


        df_last_month.unstack().plot(kind='bar', use_index=True, rot=0, 
                                     subplots=sub, title=(meas+' by '+col2))
    else:
        df_last_month = df.loc[(df[col] != num) & (df[col2] != num2)]
        df_last_month = df_last_month.groupby(by=[col, col3, col2]).agg({'Voice'+equil:meas, 
                                                                         'Equipment'+equil:meas, 
                                                                         'Data'+equil:meas})
        df_last_month.rename(columns = {'index':col3}, inplace=True)
        df_last_month = pd.DataFrame(df_last_month)
        print(df_last_month.head().T)

        df_last_month.unstack().plot(kind='bar', use_index=True, rot=0, 
                                     subplots=sub, title=(meas+' by '+col2))