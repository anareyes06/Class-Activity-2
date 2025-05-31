import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("Models/image_model.keras")

st.title("🖼 Image Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((32, 32))
    st.image(image, caption="Image uploaded", use_column_width=True)

    if st.button("Predict"):
        img_array = np.array(image) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_batch)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        predicted_label = np.argmax(prediction)
        st.write(f"Prediction: {class_names[predicted_label]} ({prediction[0][predicted_label]:.2f})")