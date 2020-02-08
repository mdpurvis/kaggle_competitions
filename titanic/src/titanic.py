import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import call
from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')

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

def quick_random_forest(df, y_col):
    x_col = list(df.columns)
    x_col.remove(y_col)
    X = df[x_col]
    y = df[[y_col]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Test Holdout Accuracy:  ", metrics.accuracy_score(y_test, y_pred))
    print("Test Holdout Recall:    ", metrics.recall_score(y_test, y_pred))
    print("Test Holdout Precision: ", metrics.precision_score(y_test, y_pred))
    print("Test Holdout F1:        ", metrics.f1_score(y_test, y_pred))
    print('\n')

    feature_imp = pd.Series(clf.feature_importances_,index=x_col).sort_values(ascending=False)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Feature Weight in Model")
    plt.show()
    print('\n')

    y_actu = pd.Series(y_test[y_col].tolist(), name='Actual')
    y_predict = pd.Series(y_pred.tolist(), name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_predict)
    print('\n',df_confusion)
    return clf
