import streamlit as st
import numpy as np
import onnxruntime as ort
def pad_sequences(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded[i] = seq[-maxlen:]
        else:
            padded[i, -len(seq):] = seq
    return padded

import json
import urllib.request
import streamlit as st


def get_word_index():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json"
    response = urllib.request.urlopen(url)
    raw = response.read().decode()
    word_index = json.loads(raw)
    return word_index

word_index = get_word_index()

# Constants
vocab_size = 10000
max_len = 500

# Load word index for decoding (optional)
word_index = imdb.get_word_index()

# Load ONNX model
@st.cache_resource
def load_model():
    return ort.InferenceSession("cnn_sentiment_analysis.onnx")

session = load_model()
def pad_sequences(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded[i] = seq[-maxlen:]
        else:
            padded[i, -len(seq):] = seq
    return padded
# Preprocess input text
def encode_text(text):
    tokens = [word_index.get(word.lower(), 2) for word in text.split()]
    padded = pad_sequences([tokens], maxlen=max_len)
    return padded.astype(np.int32)

# Streamlit UI
st.title("📚 Sentiment Analysis with CNN (ONNX)")
input_text = st.text_area("Enter a movie review to analyze sentiment:", height=150)
if st.button("Analyze"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        encoded = encode_text(input_text)
        result = session.run(None, {"input": encoded})[0]
        prob = float(result[0][0])
        label = "Positive 😊" if prob > 0.5 else "Negative 😞"
        st.success(f"Prediction: {label} (Confidence: {prob:.2f})")
