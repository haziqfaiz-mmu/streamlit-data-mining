import streamlit as st
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import imblearn
import scipy.stats as stats
from datetime import datetime 
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import scipy.stats as stats

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler() # everything will be between 0 and 1
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


def regression(df):
    st.sidebar.subheader("Chooose Regression Model")
    df_regression = df.drop(["Date"],axis=1)
    df_regression["Time"]=pd.to_datetime(df_regression["Time"], format='%H:%M:%S').dt.hour
    

    #Label encode
    label_encoder = preprocessing.LabelEncoder()
    for col in df_regression:
        if df_regression[col].dtype == "object":
            df_regression[col]=label_encoder.fit_transform(df[col])

    df_regression =stats.zscore(df_regression)

    #get the X and y
    X_reg = df_regression.drop(["humidity"],axis=1)
    y_reg = df_regression["humidity"]
    colnames = X_reg.columns

    model = st.sidebar.selectbox("Models", ("1. Linear Regression with RFE", "2. Lasso Regression with RFE", "3. 2nd Order Polynomial Regression with RFE"),key='regression-model')

    if model == "1. Linear Regression with RFE":
        lr_with_rfe(X_reg,y_reg,colnames)
       
    if model == "2. Lasso Regression with RFE":
        lasso_with_rfe(X_reg,y_reg,colnames)

    if model == "3. 2nd Order Polynomial Regression with RFE":
        poly_with_rfe(X_reg,y_reg,colnames)
        
def lr_with_rfe(X_reg,y_reg,colnames):
    rfe_features_number = st.sidebar.selectbox("How many RFE features?",(5, 10, 15,20))

    if st.sidebar.button("Run model", key='run'):
        lr = LinearRegression()
        rfe = RFECV(lr, min_features_to_select=rfe_features_number, cv=3, scoring="r2")

        rfe.fit(X_reg,y_reg)
        rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
        rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
        rfe_score = rfe_score.sort_values("Score", ascending = False)
        st.write(rfe_score)

        optimal_X = X_reg.iloc[:, rfe.support_]
        st.write(optimal_X)

        # evaluate model
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        n_scores = cross_val_score(lr,optimal_X, y_reg, cv=cv, scoring='r2')
        # report performance
        #print('R2: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        st.write('R2: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

def lasso_with_rfe(X_reg,y_reg,colnames):
    rfe_features_number = st.sidebar.selectbox("How many RFE features?",(5, 10, 15,20))
    alpha = st.sidebar.number_input('Insert alpha value')

    if st.sidebar.button("Run model", key='run'):
        lasso = Lasso(alpha=alpha)
        rfe2 = RFECV(lasso, cv=3, min_features_to_select=rfe_features_number, scoring="r2")

        rfe2.fit(X_reg,y_reg)
        rfe_score = ranking(list(map(float, rfe2.ranking_)), colnames, order=-1)
        rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
        rfe_score = rfe_score.sort_values("Score", ascending = False)
        st.write(rfe_score)

        optimal_X = X_reg.iloc[:, rfe2.support_]
        st.write(optimal_X)

        # evaluate model
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        n_scores = cross_val_score(lasso,optimal_X, y_reg, cv=cv, scoring='r2')
        # report performance
        #print('R2: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        st.write('R2: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

def poly_with_rfe(X_reg,y_reg,colnames):
    rfe_features_number = st.sidebar.selectbox("How many RFE features?",(5, 10, 15,20))

    if st.sidebar.button("Run model", key='run'):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(X_reg)

        poly_model = LinearRegression()
        rfe_poly = RFECV(poly_model, min_features_to_select=10, cv=3)
        rfe_poly.fit(poly_features,y_reg)

        rfe_score = ranking(list(map(float, rfe_poly.ranking_)), colnames, order=-1)
        rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
        rfe_score = rfe_score.sort_values("Score", ascending = False)
        st.write(rfe_score)

        pipeline = Pipeline(steps=[('s',rfe_poly),('m',poly_model)])
        # evaluate model
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        n_scores = cross_val_score(pipeline, poly_features, y_reg, cv=cv, scoring='r2')
        # report performance
        st.write('R2: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))