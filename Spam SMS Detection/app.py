import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess text using Lemmatization
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = word_tokenize(text)  # Tokenize

    y = [word for word in text if word.isalnum()]  # Remove punctuation & special characters

    # Remove stopwords
    y = [word for word in y if word not in stopwords.words('english')]

    # Lemmatization
    y = [lemmatizer.lemmatize(word) for word in y]

    return " ".join(y)

# Load the trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# 🎨 Apply Custom CSS for Better Styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            width: 60%;
            margin: auto;
        }
        .spam {
            color: white;
            background-color: red;
            padding: 10px;
            border-radius: 5px;
            font-size: 24px;
            text-align: center;
        }
        .not-spam {
            color: white;
            background-color: green;
            padding: 10px;
            border-radius: 5px;
            font-size: 24px;
            text-align: center;
        }
        .btn-custom {
            background-color: #008CBA;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            width: 100%;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# 🎯 Title Section
st.markdown('<h1 class="title">📩 Email/SMS Spam Classifier</h1>', unsafe_allow_html=True)

# 📦 Main Input Container
st.markdown('<div class="container">', unsafe_allow_html=True)

# Text input for user message
input_sms = st.text_area("✍️ Enter the message below:", height=150)

# Predict Button
if st.button('🚀 Predict', key="predict"):
    if input_sms.strip():  # Check if input is not empty
        # 1. Preprocess the text
        transformed_sms = transform_text(input_sms)
        # 2. Convert text to feature vector
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict the class
        result = model.predict(vector_input)[0]
        # 4. Display result with better styling
        if result == 1:
            st.markdown('<div class="spam">🚨 Spam Message Detected!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="not-spam">✅ This is Not Spam</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message to classify!")

# Close the container
st.markdown('</div>', unsafe_allow_html=True)
