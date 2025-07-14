import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("fake_news_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.title("📰 Fake News Detector")
text = st.text_area("Enter a news headline to check:")

if st.button("Check"):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)[0][0]
    
    label = "🟢 Real News" if pred < 0.5 else "🔴 Fake News"
    confidence = round(float(pred if pred >= 0.5 else 1 - pred) * 100, 2)

    st.subheader(f"{label} ({confidence}% confidence)")
