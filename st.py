import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from cleaning import *
from classification import *
from regression import *
from clustering import *
from download import *

def main():
    st.title("Data Mining Project")
    st.sidebar.title("Data Mining Project")
    st.sidebar.markdown("Haziq and Anwar")
    st.set_option('deprecation.showPyplotGlobalUse', False)


    @st.cache(persist=True)
    def load_data(dataset):
        df = pd.read_csv(dataset)
        df.rename(columns = {'icon':'weather'}, inplace = True)
        return df
        

    @st.cache(persist=True)
    def split(df):
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    #Load datasets
    originaldf = load_data("dataset.csv")
    weatherdf = load_data("weather.csv")
    df = load_data("merged-normalized.csv")
    
    st.sidebar.subheader("Choose Section")
    section = st.sidebar.selectbox("Sections", ("1. Data Cleaning and Exploration", "2. Classification"
    , "3. Regression", "4. Association Mining", "5. Clustering"), key='section')

    
    if section == '1. Data Cleaning and Exploration':
        data_cleaning(originaldf,df)

    if section == '2. Classification':
        classification(df)

    if section == '3. Regression':
        regression(df)

    if section == "5. Clustering":
        clustering(df)

#Add download button
    csv = convert_df(df)
    st.sidebar.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='merged-normalized.csv',
    mime='text/csv',)

    with open("test.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()
    st.sidebar.download_button(
    label="Download report as PDF",
    data=PDFbyte,
    file_name='test.pdf',
    mime='application/octet-stream',
)

if __name__ == '__main__':
    main()
