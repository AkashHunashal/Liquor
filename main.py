from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from datetime import datetime
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
import scipy.stats as stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def readdata():
    data = pd.read_csv('liquor.csv')
    data.drop('convenience_store', axis=1, inplace=True)
    data.dropna(axis=0, inplace=True)
    data.drop(data[(data['total'] > 10000)].index, axis=0, inplace=True)
    data = data.reset_index()
    return data


unique_categories = []


def preprocessing(data):
    data['date'] = pd.to_datetime(data['date'])
    data['weekday'] = data['date'].dt.day_name()
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data.drop('date', axis=1, inplace=True)
    for column in data.iloc[:, -3:]:
        unique_categories.append({column: sorted(data[column].unique())})
    numerical_features = data.select_dtypes(include=[int, float]).columns
    categorical_features = data.select_dtypes(exclude=[int, float]).columns
    data2 = data.copy()
    global categorical_encoded
    categorical_encoded = []
    for categories in categorical_features:
        category_dict = data2.groupby(categories)['total'].mean(
        ).sort_values(ascending=False).to_dict()
        data2[categories+'_Enc'] = data2[categories].map(category_dict)
        category_encoded = {categories: category_dict}
        categorical_encoded.append(category_encoded)
    numerical_features = numerical_features.drop('total')
    threshold = 0.7
    corr_list = []
    for column1 in numerical_features:
        for column2 in numerical_features:
            if column1 != column2:
                if (abs(round(data2[[column1, column2]].corr(), 2)) >= threshold).iloc[1:, :1].values:
                    flag = 1
                    if flag == 1:
                        numerical_features = numerical_features.drop(column1)
                        corr_list.append(column1)
    data2.drop(corr_list, inplace=True, axis=1)
    data2.drop(labels=categorical_features, axis=1, inplace=True)
    for categories in data2.iloc[:, :8]:
        category_dict = data2.groupby(categories)['total'].mean(
        ).sort_values(ascending=False).to_dict()
        data2[categories+'_Enc'] = data2[categories].map(category_dict)
        category_encoded = {categories: category_dict}
        categorical_encoded.append(category_encoded)
    data2.drop(data2.columns[:8], axis=1, inplace=True)
    for categories in data2.iloc[:, 1:4]:
        category_dict = data2[categories].value_counts(
        ).sort_values(ascending=False).to_dict()
        data2[categories+'_Enc'] = data2[categories].map(category_dict)
        category_encoded = {categories: category_dict}
        categorical_encoded.append(category_encoded)
    data2.drop(data2.columns[1:4], axis=1, inplace=True)
    return data2


def modelbuild(data2):
    X = data2.drop('total', axis=1)
    y = data2[['total']]
    corr_set = set()
    threshold = 0.7
    for categories in X.columns:
        for i in X.columns:
            if categories != i:
                if abs(round(X[[categories, i]].corr(), 2) > threshold).values[0][1]:
                    corr_set.add(categories)
    X.drop(corr_set, axis=1, inplace=True)
    for column in X.columns:
        X[column] = np.log1p(X[[column]])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def modelprediction(X_test):
    xgboost_model = pickle.load(open('XGBoost_regression_model.pkl', 'rb'))
    xgboost_pred = xgboost_model.predict(X_test)
    return xgboost_pred


def trainmodel():
    data = readdata()
    data = preprocessing(data)
    X_train, X_test, y_train, y_test = modelbuild(data)
    prediction = modelprediction(X_test)
    return prediction


def testmodel(data):
    trainmodel()
    for column in data.columns:
        for category in categorical_encoded:
            for item, values in category.items():
                if item == column:
                    data[column+'_Enc'] = data[column].map(values)
    data.drop(data.columns[0:6], axis=1, inplace=True)
    for column in data.columns:
        data[column] = np.log1p(data[[column]])
    xgboost_model = pickle.load(open('XGBoost_regression_model.pkl', 'rb'))
    xgboost_pred = xgboost_model.predict(data)
    return xgboost_pred
