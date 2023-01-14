import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def data_cleaning(originaldf,df):
    st.markdown("Original Dataset")
    st.dataframe(originaldf)
    st.markdown("Merged, Cleaned and Normalized Dataset")
    st.dataframe(df)

    categorical_col = df.select_dtypes(exclude='number').columns
    categorical_col = categorical_col[1:]
    categorical_plot=st.selectbox("View Bar Chart of Categorical Variables",(categorical_col),key="categorical columns")
    fig = plt.figure() 
    df[categorical_plot].value_counts().plot(kind='bar')
    st.pyplot(fig)
    numerical_col = df.select_dtypes(include='number').columns
    numerical_plot=st.selectbox("View Histogram of Numercial Variables",(numerical_col),key="numerical columns")
    fig2 = plt.figure() 
    df[numerical_plot].plot(kind='hist')
    st.pyplot(fig2)