import tensorflow as tf
import numpy as np
import pickle
import os

# Cargar el tokenizer previamente guardado (ajusta la ruta si es necesario)
TOKENIZER_PATH = os.path.join("Models", "tokenizer.pkl")

tokenizer = None
if os.path.exists(TOKENIZER_PATH):
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
else:
    print(f"[ERROR] Tokenizer file not found at {TOKENIZER_PATH}. Please ensure the file exists.")

MAX_LEN = 100  # Ajusta este valor al usado durante el entrenamiento

def preprocess_text(text):
    """
    Preprocesa el texto usando el tokenizer entrenado y lo convierte en una secuencia adecuada para el modelo.
    """
    if tokenizer is None:
        raise FileNotFoundError(f"Tokenizer file not found at {TOKENIZER_PATH}. Please ensure the file exists.")
    # Opcional: limpieza básica
    text = text.lower().strip()
    # Tokenización y padding
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded[0]