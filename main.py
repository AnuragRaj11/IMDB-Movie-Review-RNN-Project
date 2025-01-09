import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset and retrieve the word index
(train_data, train_labels), (test_data, test_labels) = imdb.load_data()
word_index = imdb.get_word_index()

# Reverse the word index to map indices back to words
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simplernnimdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # Adjusted index
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review): 
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input) 
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative' 
    return sentiment, prediction[0][0]

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as Positive or Negative.')

user_input = st.text_area('Movie Review', height=150)

if st.button('Classify'):
    if user_input.strip():  
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        st.write(f'Sentiment: **{sentiment}**')
        st.write(f'Prediction Score: {prediction[0][0]:.2f}')
    else:
        st.write('Please enter a valid movie review.')
else:
    st.write('Click the "Classify" button to analyze the sentiment.')
