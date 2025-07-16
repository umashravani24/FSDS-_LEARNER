import streamlit as st
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
tips=sns.load_dataset("tips")
st.title("Seaborn Visualization Workshop")
st.write("This is a workshop to explore various Seaborn visualizations using the tips dataset.")
