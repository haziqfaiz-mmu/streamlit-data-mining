import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def kmeans(df):
    st.markdown("Perform Clustering using K-Means CLustering")

    df = pd.read_csv("merged-normalized.csv")
    df = df.drop(["Date"],axis=1)
    df["Time"]=pd.to_datetime(df["Time"], format='%H:%M:%S').dt.hour

    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    sns.scatterplot(x="tempmin", y="humidity", data=df, ax=axes[0])
    sns.scatterplot(x="tempmax", y="humidity", data=df, ax=axes[1])

    st.markdown("Scatterplots to observe the relationship between temperature and humidity. We can see that we can divide these into vertical clusters.")
    st.pyplot(fig)

    #Label encode
    label_encoder = preprocessing.LabelEncoder()
    for col in df:
        if df[col].dtype == "object":
            df[col]=label_encoder.fit_transform(df[col])
    
    df_scale=pd.DataFrame(StandardScaler().fit_transform(df))

    sns.heatmap(df_scale.corr())
    st.write("The data have very weak correlation with each other so dividing them into clusters might be tough.")

    sns.heatmap(df_scale.corr())

    st.pyplot()