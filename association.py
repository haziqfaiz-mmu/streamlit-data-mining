import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def association(df):
    st.header("Apriori Algorithm")
    if st.sidebar.button("Show Results", key='run'):
        
        association_results = np.load('association_results.npy',allow_pickle=True)
        cnt = 0
        for item in association_results:
            cnt += 1
            # first index of the inner list
            # Contains base item and add item
            pair = item[0] 
            items = [x for x in pair]
            st.write("(Rule " , str(cnt) + ") " , items[0] , " -> " , items[1])

            #second index of the inner list
            st.write("Support: " , str(round(item[1],3)))

            #third index of the list located at 0th
            #of the third index of the inner list

            st.write("Confidence: " , str(round(item[2][0][2],4)))
            st.write("Lift: " , str(round(item[2][0][3],4)))
            st.write("=====================================") 