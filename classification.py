import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime 
import random

from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter

import statsmodels.api as sm
from statsmodels.formula.api import ols

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 50)
pd.set_option('display.max_rows', 50)

from apyori import apriori
from boruta import BorutaPy
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime 
import random

import statsmodels.api as sm
from statsmodels.formula.api import ols

from collections import Counter
from sklearn.impute import SimpleImputer
from tqdm import tqdm 
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 50)
pd.set_option('display.max_rows', 50)

def classification(df):
    st.title("Classification")
    st.sidebar.subheader("Chooose Classification Model")
    model = st.sidebar.selectbox("Models", ("1. Naive Bayes", "2. Random Forest with Boruta", "3. Stacking Ensemble"),key='regression-model')

    df_ori = pd.read_csv("merged-normalized.csv")
    df_drop = df_ori.drop(['Date', 'Time', 'latitude', 'longitude'], axis=1)
    df_label = df_drop.copy()

    label_encoder = preprocessing.LabelEncoder()

    for col in df_label:
        if df_label[col].dtype == "object":
            df_label[col]=label_encoder.fit_transform(df_label[col])

    # for Basket_Size
    y = df_label['Basket_Size']
    X = df_label.drop(['Basket_Size'], axis=1)
    colnames = X.columns

    if model == "1. Naive Bayes":
        naivebayes(df_label,X,y,colnames)
    if model ==  "2. Random Forest with Boruta":
        randomforest(df_label,X,y,colnames)
    if model == "3. Stacking Ensemble":
        stacking(df_label,X,y)



def ranking(ranks, names, order=1):
    minmax = MinMaxScaler() # everything will be between 0 and 1
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

def naivebayes(df_label,X,y,colnames):
    if st.sidebar.button("Run model", key='run'):
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced_subsample', max_depth=5)

        feat_selector_rf = BorutaPy(rf, n_estimators='auto', random_state=1)
        feat_selector_rf.fit(X.values, y.values.ravel())

        boruta_score = ranking(list(map(float, feat_selector_rf.ranking_)), colnames, order=-1)
        boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
        boruta_score = boruta_score.sort_values("Score", ascending = False)

        st.write("The top 10 BORUTA features are:")
        st.write(boruta_score.head(10))
        st.write("The bottom 10 BORUTA features are")
        st.write(boruta_score.tail(10))

        head = boruta_score.head(15) 
        head_features = head['Features']

        y = df_label['Basket_Size']
        X = df_label[head_features] 
        colnames = X.columns

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

        nb = GaussianNB()
        nb.fit(X_train, y_train)

        y_pred = nb.predict(X_test)

        st.write("The accuracy for naive bayes is",nb.score(X_test, y_test))

def randomforest(df_label,X,y,colnames):
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced_subsample', max_depth=5)

        feat_selector_rf = BorutaPy(rf, n_estimators='auto', random_state=1)
        feat_selector_rf.fit(X.values, y.values.ravel())

        boruta_score = ranking(list(map(float, feat_selector_rf.ranking_)), colnames, order=-1)
        boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
        boruta_score = boruta_score.sort_values("Score", ascending = False)

        st.write("The top 10 BORUTA features are:")
        st.write(boruta_score.head(10))
        st.write("The bottom 10 BORUTA features are")
        st.write(boruta_score.tail(10))

        head = boruta_score.head(15) 
        head_features = head['Features']

        y = df_label['Basket_Size']
        X = df_label[head_features] 
        colnames = X.columns

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

        nb = GaussianNB()
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        st.write("The accuracy for random forest is",rf.score(X_test, y_test))

def stacking(df_label,X,y,head_features):
    models = dict()

    models["knn"] = KNeighborsClassifier()
    models["cart"] = DecisionTreeClassifier()
    models["rf"] = RandomForestClassifier()
    models["bayes"] = GaussianNB()

    models["stacking"] = get_stacking()

    y = df_label['Basket_Size']
    X = df_label[head_features] 
    colnames = X.columns
    feat_selector_rf = BorutaPy(rf, n_estimators='auto', random_state=1)
    feat_selector_rf.fit(X.values, y.values.ravel())

    boruta_score = ranking(list(map(float, feat_selector_rf.ranking_)), colnames, order=-1)
    boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
    boruta_score = boruta_score.sort_values("Score", ascending = False)

    st.write("The top 10 BORUTA features are:")
    st.write(boruta_score.head(10))
    st.write("The bottom 10 BORUTA features are")
    st.write(boruta_score.tail(10))

    head = boruta_score.head(15) 
    head_features = head['Features']

    # Evaluate models and store results
    results, names = list(), list()

    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        print(">%s %.3f (%.3f)" % (name, mean(scores), std(scores)))
    
    # plot model performance for comparison based on f1-score
    plt.boxplot(results, labels=names, showmeans=True)
    plt.show()

# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('cart', DecisionTreeClassifier()))
    level0.append(('rf', RandomForestClassifier()))    
    level0.append(('bayes', GaussianNB()))
    
    # define stacking ensemble
    level1 = GaussianNB()     
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)   
    
    return model

# evaluate model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=10)
    scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')
    return scores


    