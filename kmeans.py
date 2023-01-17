import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from yellowbrick.cluster import silhouette_visualizer

def kmeans(df):
    st.title("Clustering")
    st.markdown("Perform Clustering using K-Means CLustering")

    df = pd.read_csv("merged-normalized.csv")
    df = df.drop(["Date"],axis=1)
    df["Time"]=pd.to_datetime(df["Time"], format='%H:%M:%S').dt.hour

    fig, axes = plt.subplots(1, 2)
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
    fig = plt.figure()
    sns.heatmap(df_scale.corr())
    st.pyplot(fig)

    #initialize kmeans parameters
    kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "random_state": 1,
    }

    #create list to hold SSE values for each k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df_scale)
        sse.append(kmeans.inertia_)

    #visualize results
    fig=plt.figure()
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Distortion")
    st.markdown("Find the best number of k for k-means")
    st.pyplot(fig)

    #instantiate the k-means class, using optimal number of clusters
    kmeans = KMeans(init="random", n_clusters=10, n_init=10, random_state=1)

    #fit k-means algorithm to data
    kmeans.fit(df_scale)
    df['cluster'] = kmeans.labels_

    fig, axes = plt.subplots(1, 2, figsize=(12,6))

    sns.scatterplot(x="tempmin", y="humidity", data=df, ax=axes[0], hue='cluster')
    sns.scatterplot(x="tempmax", y="humidity", data=df, ax=axes[1],hue='cluster')

    st.markdown("Perform k-means with 10 ckusters and analyze the silhouette plot")

    st.pyplot(fig)

    st.markdown("Our silhouette plot show that each cluster has a roughly similar thickness and a lot of points are above the average silhouette score. We conclude that we achieved a good clustering.")
    fig, ax = plt.subplots()
    silhouette_visualizer(KMeans(10, random_state = 12), df, color = 'yellowbricks',ax=ax)
    st.pyplot(fig)
