import streamlit as st
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("Models/regression_model.keras")

st.title("📈 Regression Model")

inputs = []
for i in range(13):
    val = st.number_input(f"Value {i+1}", key=f"val_{i+1}")
    inputs.append(val)

if st.button("Predict"):
    X = np.array([inputs])
    pred = model.predict(X)
    st.write(f"Model Prediction: {pred[0][0]:.2f}")