import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle

# Load the saved model (from .h5 file)
model = tf.keras.models.load_model('emotion_recognizer.h5')

# Load the saved tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Maximum length of the input text (same as used during model training)
maxlen = 40  # Adjust this according to your model training settings

# Emotion classes (from 0 to 4)
emotion_classes = ["Anger", "Joy", "Love", "Sadness", "Surprise"]

# Preprocessing the input text
def preprocess_text(text):
    # Tokenize and remove stopwords and punctuation
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = word_tokenize(text)
    filtered_text = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
    return ' '.join(filtered_text)

# Function to predict emotion
def predict_emotion(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Convert the text to a sequence of tokens using the tokenizer
    seq = tokenizer.texts_to_sequences([preprocessed_text])
    
    # Pad the sequences to the same length used during training
    padded_seq = pad_sequences(seq, maxlen=maxlen)
    
    # Predict the emotion
    pred = model.predict(padded_seq)
    pred_class = np.argmax(pred, axis=1)[0]
    
    # Return the predicted emotion
    return emotion_classes[pred_class]

# Main function to handle input and output
if __name__ == "__main__":
    # Sample text input (can be modified to take dynamic input)
    input_text = input("Enter a text to predict the emotion: ")
    
    # Predict emotion for the input text
    predicted_emotion = predict_emotion(input_text)
    
    print(f"Predicted Emotion: {predicted_emotion}")
