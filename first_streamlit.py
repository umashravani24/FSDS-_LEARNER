
import streamlit as st
st.title(" first streamlit app created by UMASHRAVANI")
st.write("welcome this app calculates square of numbers")
st.header("SELECT NUMBER")
n=st.slider("pic a number",0,100,25)
st.subheader("result")
square_num=n*n
st.write(f"the square of**{ n }** is  **{square_num}**")

