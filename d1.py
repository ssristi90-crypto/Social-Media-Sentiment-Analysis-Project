# ---------------------------
# STEP 2: DATA PREPROCESSING
# ---------------------------

# Import libraries
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ---------------------------
# Load the Sentiment140 dataset
# ---------------------------
# Sentiment140 CSV usually has columns: target, ids, date, flag, user, text
# (target: 0 = negative, 4 = positive)
df = pd.read_csv("sentiment140.csv", encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Convert target to readable labels
df['sentiment'] = df['target'].replace({0: 'negative', 4: 'positive'})
df = df[['text', 'sentiment']]  # Keep only necessary columns

print("Original data sample:")
print(df.head())

# ---------------------------
# Define cleaning functions
# ---------------------------

# 1️⃣ Basic text cleaning (remove URLs, mentions, hashtags, punctuation, numbers)
def clean_text(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # remove mentions & hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# 2️⃣ Remove stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

# 3️⃣ Lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)

# ---------------------------
# Apply preprocessing steps
# ---------------------------
print("\nCleaning text... (this may take a few minutes for large datasets)")

df['clean_text'] = df['text'].apply(clean_text)
df['clean_text'] = df['clean_text'].apply(remove_stopwords)
df['clean_text'] = df['clean_text'].apply(lemmatize_text)

# ---------------------------
# Check results
# ---------------------------
print("\nCleaned data sample:")
print(df[['text', 'clean_text', 'sentiment']].head())

# Optional: save the cleaned dataset
df.to_csv("cleaned_sentiment140.csv", index=False)

print("\n✅ Data preprocessing completed and saved as 'cleaned_sentiment140.csv'")

# Assuming your DataFrame 'df' now has a 'clean_text' column

# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Load the cleaned dataset (if you restarted your environment)
# df = pd.read_csv("cleaned_sentiment140.csv") 

# 1. Define Features (X) and Target (y)
X = df['clean_text']
# Convert 'sentiment' column to numerical labels (0 and 1)
# NOTE: If you saved the file and reloaded it, ensure 'sentiment' is mapped correctly
y = df['sentiment'].replace({'negative': 0, 'positive': 1})

print("\nStarting Feature Extraction (TF-IDF) and Data Split...")

# 2. Split Data into Training and Testing Sets (20% for testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Initialize and Fit the TF-IDF Vectorizer
# Limits vocabulary to the 50,000 most common words
tfidf_vectorizer = TfidfVectorizer(max_features=50000)

# Fit and transform the TRAINING data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the TEST data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"✅ Data split and TF-IDF features created. Vocabulary size: {X_train_tfidf.shape[1]}")

# 4. Train the Baseline Model (Logistic Regression)
print("\nStarting Baseline Modeling (Logistic Regression)...")
log_reg_model = LogisticRegression(
    max_iter=1000, 
    solver='liblinear',
    random_state=42,
    n_jobs=-1 
)

log_reg_model.fit(X_train_tfidf, y_train)
print("✅ Model Training Complete.")

# 5. Evaluate the Model (Step 5 in your plan)
y_pred = log_reg_model.predict(X_test_tfidf)

print("\n--- Model Evaluation (Baseline: Logistic Regression) ---")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# STEP 4: ADVANCED MODELING (LSTM)
# ---------------------------

# Install if you haven't already: pip install tensorflow

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Assume X (clean_text) and y (0/1 labels) are ready from the previous steps
# Assume X_train, X_test, y_train, y_test are ready

# --- 1. Prepare Data for Deep Learning ---

# Define hyperparameters
MAX_WORDS = 50000 
MAX_SEQUENCE_LENGTH = 50 # Max length of a tweet sequence
EMBEDDING_DIM = 100

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<oov>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences of integers
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences (make all tweets the same length)
X_train_padded = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test_padded = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

print(f"✅ Data for LSTM prepared. Sequence shape: {X_train_padded.shape}")

# --- 2. Build the LSTM Model ---

model = Sequential([
    # Input: MAX_WORDS, Output: EMBEDDING_DIM
    Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    # LSTM layer processes the sequence
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    # Output layer for binary classification (Sigmoid)
    Dense(1, activation='sigmoid') 
])

# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

# --- 3. Train the Model ---
print("\nStarting LSTM Model Training...")

history = model.fit(
    X_train_padded, 
    y_train, 
    epochs=5, # Start with a low number, Deep Learning trains slowly
    batch_size=512,
    validation_data=(X_test_padded, y_test)
)

print("✅ LSTM Model Training Complete.")