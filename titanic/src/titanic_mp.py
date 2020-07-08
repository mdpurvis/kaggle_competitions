# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:06:43 2020

@author: mdpur
"""

import pandas as pd


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
    
       
    
    
#Creates Histogram for Comparitive Purposes
def review_detail(df, col, num, col2, num2, sub, meas, type1, stack):    
    if type1 == 'all':
        df_last_month = df.loc[(df[col] != num) & (df[col2] == num2)]
        df_last_month = df_last_month.groupby(by=[col, col2, 'Survived']).agg({'Survived':meas})
        df_last_month.rename(columns = {'index':'index'}, inplace=True)
        df_last_month['pct_total'] = ((df_last_month['Survived'] / df_last_month['Survived'].sum())*100).astype(int)
        df_last_month = pd.DataFrame(df_last_month)
        df_last_month = df_last_month.T
        print(df_last_month)
        df_last_month.drop_duplicates(inplace=True)
        df_last_month = df_last_month.T
        df_last_month.drop(columns=['pct_total'], inplace=True)


        df_last_month.unstack().plot(kind='bar', use_index=True, rot=0, 
                                     subplots=sub, stacked=stack)       
    else:
        df_last_month = df.loc[(df[col] != num) & (df[col2] != num2)]
        df_last_month = df_last_month.groupby(by=[col, col2, 'Survived']).agg({'Survived':meas})
        df_last_month.rename(columns = {'index':'index'}, inplace=True)
        df_last_month['pct_total'] = ((df_last_month['Survived'] / df_last_month['Survived'].sum())*100).astype(int)
        df_last_month = pd.DataFrame(df_last_month)
        df_last_month = df_last_month.T
        print(df_last_month)
        df_last_month.drop_duplicates(inplace=True)
        df_last_month = df_last_month.T
        df_last_month.drop(columns=['pct_total'], inplace=True)


        df_last_month.unstack().plot(kind='bar', use_index=True, rot=0, 
                                     subplots=sub, stacked=stack)       