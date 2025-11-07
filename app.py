# ---------------------------
# app.py (The complete Sentiment Analysis Dashboard)
# ---------------------------
import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- SETUP: Load Tools ---

# NOTE: You may need to run these downloads in your terminal once if the app fails due to missing NLTK data:
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


# Load the saved model and vectorizer
try:
    # Ensure these files are in the same folder as app.py
    model = joblib.load('log_reg_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or Vectorizer files not found. Ensure 'log_reg_model.pkl' and 'tfidf_vectorizer.pkl' are in the same folder as app.py.")
    st.stop()


# --- PREPROCESSING FUNCTIONS (MUST be identical to training) ---
# Define cleaning functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions & hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def remove_stopwords(text):
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

def lemmatize_text(text):
    tokens = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)

def preprocess_text_for_prediction(raw_text):
    """Applies the full cleaning pipeline."""
    cleaned = clean_text(raw_text)
    cleaned = remove_stopwords(cleaned)
    final_text = lemmatize_text(cleaned)
    return final_text

# --- PREDICTION FUNCTION ---

def predict_sentiment(raw_text):
    # 1. Preprocess the raw input text
    processed_text = preprocess_text_for_prediction(raw_text)
    
    # 2. Transform the text using the fitted vectorizer
    # NOTE: The input must be a list containing the text, hence [processed_text]
    text_vector = vectorizer.transform([processed_text])
    
    # 3. Predict (0 for Negative, 1 for Positive)
    prediction = model.predict(text_vector)[0]
    
    # 4. Return result
    return 'POSITIVE' if prediction == 1 else 'NEGATIVE'

# --- Streamlit App Interface ---
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("💬 Social Media Sentiment Analyzer")
st.markdown("---")

st.header("Analyze Text Sentiment")
user_input = st.text_area("Enter a tweet or text snippet below:", height=150)

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner('Analyzing sentiment...'):
            result = predict_sentiment(user_input)
            
            st.markdown("### Prediction Result:")
            if result == 'POSITIVE':
                st.success(f"Result: **{result}** 😄")
            else:
                st.error(f"Result: **{result}** 😠")
            st.markdown(f"*(Model is Logistic Regression trained on 1.6M tweets with **~78% accuracy**)*")
    else:
        st.warning("Please enter some text for analysis.")