import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def classification(df):
    st.sidebar.subheader("Chooose Classification Model")
    model = st.sidebar.selectbox("Models", ("1. Naive Bayes", "2. Support Vector Machines"),key='classification-model')