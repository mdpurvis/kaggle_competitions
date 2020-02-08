import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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

def split_cabin(df):
    df.loc[df['Cabin'].notnull(), 'Cabin Prefix'] = df.loc[
        df['Cabin'].notnull(), 'Cabin'].str[0]
    df.loc[df['Cabin'].notnull(), 'Cabin Number'] = df.loc[
        df['Cabin'].notnull(), 'Cabin'].str.extract('(\d+)', expand=False)
    df['Cabin Number'] = pd.to_numeric(df['Cabin Number']).astype('Int64')
    return df

def append_one_hot(df, col):
    df['Embarked'] = df['Embarked'].fillna('')
    ohe = OneHotEncoder(sparse = False, dtype = int, handle_unknown = 'ignore')
    data = ohe.fit_transform(df[['Embarked']])

    col_names = [col + ' ' + val for val in list(ohe.categories_)[0]]

    temp_df = pd.DataFrame(data, columns = col_names, index = df.index)
    temp_df.drop(col + ' ', axis=1, inplace=True)
    df = df.join(temp_df)
    return df
