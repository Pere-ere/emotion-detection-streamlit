import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# ----------------- Load Model from Hugging Face -----------------
MODEL_NAME = "Pere-ere/fine_tuned_bert_film_sentiment_modell"

@st.cache_resource
def load_model():
    """Loads the fine-tuned model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return model, tokenizer

model, tokenizer = load_model()

# ----------------- Emotion Labels -----------------
emotion_labels = {
    0: ("Sadness", "😢"),
    1: ("Joy", "😃"),
    2: ("Love", "❤️"),
    3: ("Anger", "😡"),
    4: ("Fear", "😨"),
    5: ("Surprise", "😲")
}

# ----------------- Emotion Prediction Function -----------------
def predict_emotion(text):
    """Predicts emotion for input text and returns label & probabilities."""
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    predictions = model(inputs)
    logits = predictions.logits.numpy()
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    predicted_class = np.argmax(probs)
    emotion_name, emoji = emotion_labels[predicted_class]
    return (emotion_name, emoji), probs

# ----------------- Visualization Function -----------------
def plot_emotion_probabilities(probs):
    """Displays emotion probability distribution as a bar chart."""
    st.markdown("### Emotion Probability Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = ["blue", "gold", "red", "darkred", "purple", "orange"]
    emotion_names = [v[0] for v in emotion_labels.values()]
    
    ax.bar(emotion_names, probs, color=colors)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_xlabel("Emotion", fontsize=12)
    ax.set_xticklabels(emotion_names, rotation=45, fontsize=10)
    
    st.pyplot(fig)
