import streamlit as st
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import os

# Download necessary NLTK data. This is crucial for Streamlit Cloud.
# This check prevents repeated downloads on each rerun
try:
    _stopwords = stopwords.words('english')
except LookupError:
    st.write("Downloading NLTK stopwords data...")
    nltk.download('stopwords')
    _stopwords = stopwords.words('english')

STOPWORDS = set(_stopwords)
stemmer = PorterStemmer()

# Function to preprocess text
def preprocess_text(text):
    # Handle potential non-string inputs gracefully
    if not isinstance(text, str):
        return ""
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return ' '.join(review)

# Define the path to the Models directory
MODELS_DIR = 'Models'

# Load the CountVectorizer, Scaler, and Model
# Use st.cache_resource to cache these objects across runs
@st.cache_resource
def load_model_files():
    cv_path = os.path.join(MODELS_DIR, 'countVectorizer.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    model_path = os.path.join(MODELS_DIR, 'random_forest_model.pkl') # Assuming you saved your model as 'random_forest_model.pkl'

    if not os.path.exists(cv_path):
         raise FileNotFoundError(f"CountVectorizer file not found at {cv_path}")
    if not os.path.exists(scaler_path):
         raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Model file not found at {model_path}")

    with open(cv_path, 'rb') as f:
        cv = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return cv, scaler, model

try:
    cv, scaler, model = load_model_files()
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}. Please ensure the '{MODELS_DIR}' directory with the necessary files is in your repository.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading model files: {e}")
    st.stop()


st.title("Amazon Alexa Review Sentiment Analysis")
st.write("Enter a review to predict its sentiment (Positive/Negative).")

# Text input from user
user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess the input
        processed_input = preprocess_text(user_input)

        # Vectorize the input using the loaded CountVectorizer
        # Need to handle potential new words not in the original vocabulary
        input_vector = cv.transform([processed_input]).toarray()

        # Scale the vectorized input using the loaded Scaler
        input_scaled = scaler.transform(input_vector)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Display prediction
        if prediction[0] == 1:
            st.success("Prediction: Positive Feedback")
        else:
            st.error("Prediction: Negative Feedback")
    else:
        st.warning("Please enter a review to predict.")
