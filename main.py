import streamlit as st

st.set_page_config(page_title="Deep Learning App", layout="wide")

st.title("Welcome to the Deep Learning App")
st.write("This application allows you to test three deep learning models:")
st.markdown("""
- 📄 Text Classification
- 🖼 Image Classification
- 📈 Regression
""")

st.write("Use the sidebar menu to navigate between the pages.")