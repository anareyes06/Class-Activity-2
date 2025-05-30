import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# model 
model = tf.keras.models.load_model("Models/text_model.keras")

st.title("📄 Text Classification")

user_input = st.text_area("Write your text here")

if st.button("Classify"):
    word_index = imdb.get_word_index()
    def text_to_sequence(text):
        words = text.lower().split()
        return [word_index.get(word, 2) + 3 for word in words]

    sequence = text_to_sequence(user_input)
    padded = pad_sequences([sequence], maxlen=500)
    pred = model.predict(padded)
    st.write(f"Prediction: {'Positive' if pred[0][0] >= 0.5 else 'Negative'} ({pred[0][0]:.2f})")