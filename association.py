import streamlit as st
import matplotlib.pyplot as plt

def association(df):
    if st.sidebar.button("Run model", key='run'):
        print("ass")