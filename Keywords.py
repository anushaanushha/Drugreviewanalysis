import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained sentiment model
with open("sentiment_model.pkl", "rb") as file:
    pipeline = pickle.load(file)

vectorizer = pipeline.named_steps["tfidf"]  # Extract TF-IDF vectorizer

# Function to predict sentiment
def predict_sentiment(review):
    sentiment_label = pipeline.predict([review])[0]
    sentiment_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    return sentiment_map[sentiment_label]

# Function to extract top N keywords (without repetition)
def extract_keywords(text, top_n=3):
    X = vectorizer.transform([text])  # Transform input text
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = X.toarray()[0]
    sorted_indices = np.argsort(tfidf_scores)[-top_n:]  # Get top N indices

    keywords = [feature_array[i] for i in reversed(sorted_indices) if tfidf_scores[i] > 0]

    # Remove redundant words if already in a longer phrase
    final_keywords = []
    for kw in keywords:
        if not any(kw in longer_kw and kw != longer_kw for longer_kw in keywords):
            final_keywords.append(kw)

    return final_keywords
