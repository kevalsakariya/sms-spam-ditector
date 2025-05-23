# RUN CODE = streamlit run "g:\python_pycharm\machine learning\project\sms spam ditector\app.py"

import streamlit as st
import pickle
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.utils.validation import check_is_fitted

# Download required NLTK datasets
nltk.download('stopwords')
nltk.download('punkt')



# Initialize the stemmer
ps = PorterStemmer()

# Text Preprocessing Function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]  # Remove special characters

    stop_words = set(stopwords.words('english'))  # Load stopwords once
    text = [i for i in y if i not in stop_words and i not in string.punctuation]

    text = [ps.stem(i) for i in text]  # Apply stemming

    return " ".join(text)

# Load Model and Vectorizer Safely
model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Error: Model or Vectorizer file is missing! Run `train_model.py` first.")
    st.stop()

try:
    with open(vectorizer_path, 'rb') as f:
        tfidf = pickle.load(f)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    check_is_fitted(model)  #  Ensure the model is fitted
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        # 1. Preprocess the text
        transformed_sms = transform_text(input_sms)

        # 2. Convert to vector form
        try:
            vector_input = tfidf.transform([transformed_sms])
        except Exception as e:
            st.error(f"Error during vectorization: {e}")
            st.stop()

        # 3. Make prediction
        try:
            result = model.predict(vector_input)[0]
            if result == 0:
                st.header("Not Spam")
            else:
                st.header("Spam Message!")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            
