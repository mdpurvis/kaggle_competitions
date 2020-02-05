import pandas as pd
import numpy as np

def identify_blanks(df):
    t = len(df)
    for each in df.columns:
        i = len(df.loc[df[each].isnull()])
        pct = int((i/t)*100)
        print(f'column {each} has {i} missing values.  {pct}%')

def one_hot_age_gender(df):
    df['Boy'] = 0
    df['Girl'] = 0
    df['Man'] = 0
    df['Woman'] = 0

    df.loc[((df['Boy'] == 0) & (df['Girl'] == 0) &  (df['Man'] == 0) & (df['Woman'] == 0)) &
           (df['Age'] < 16) & (df['Sex'] == 'male'), ['Boy']] = 1
    df.loc[((df['Boy'] == 0) & (df['Girl'] == 0) &  (df['Man'] == 0) & (df['Woman'] == 0)) &
           (df['Age'] < 16) & (df['Sex'] == 'female'), ['Girl']] = 1
    df.loc[((df['Boy'] == 0) & (df['Girl'] == 0) &  (df['Man'] == 0) & (df['Woman'] == 0)) &
           (df['Age'] >= 16) & (df['Sex'] == 'male'), ['Man']] = 1
    df.loc[((df['Boy'] == 0) & (df['Girl'] == 0) &  (df['Man'] == 0) & (df['Woman'] == 0)) &
           (df['Age'] >= 16) & (df['Sex'] == 'female'), ['Woman']] = 1

    df.loc[((df['Boy'] == 0) & (df['Girl'] == 0) &  (df['Man'] == 0) & (df['Woman'] == 0)) &
           (df['Name'].str.contains('Mr.') & (df['Sex'] == 'male')), ['Man']] = 1
    df.loc[((df['Boy'] == 0) & (df['Girl'] == 0) &  (df['Man'] == 0) & (df['Woman'] == 0)) &
           (df['Name'].str.contains('Mrs.') & (df['Sex'] == 'female')), ['Woman']] = 1

    df.loc[((df['Boy'] == 0) & (df['Girl'] == 0) &  (df['Man'] == 0) & (df['Woman'] == 0)) &
           (df['Name'].str.contains('Master.') & (df['Sex'] == 'male')), ['Boy']] = 1
    df.loc[((df['Boy'] == 0) & (df['Girl'] == 0) &  (df['Man'] == 0) & (df['Woman'] == 0)) &
           (df['Name'].str.contains('Miss.') & (df['Sex'] == 'female')), ['Girl']] = 1

    df.loc[((df['Boy'] == 0) & (df['Girl'] == 0) &  (df['Man'] == 0) & (df['Woman'] == 0)) &
           (df['Name'].str.contains('Dr.') & (df['Sex'] == 'male')), ['Man']] = 1

    return df
