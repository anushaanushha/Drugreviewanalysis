import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import pickle

# Download stopwords if not available
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load dataset
df = pd.read_csv("drugsComTrain_raw.csv")  # Update with actual dataset path
df.dropna(subset=["review", "condition", "rating", "drugName"], inplace=True)

# Load trained model and vectorizer
model = pickle.load(open("lisentiment_model_lgb.pkl", "rb"))
vectorizer = pickle.load(open("litfidf_vectorizer_lgb.pkl", "rb"))

# List of unique drug names from the dataset
drug_names = sorted(df["drugName"].dropna().unique().tolist())

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Function to classify sentiment using trained model
def classify_review(review):
    cleaned_review = clean_text(review)
    X_input = vectorizer.transform([cleaned_review])
    prediction = model.predict(X_input)
    sentiments = ["negative", "neutral", "positive"]
    return sentiments[int(prediction[0])] if prediction.ndim == 1 else sentiments[np.argmax(prediction, axis=1)[0]]

# Function to analyze sentiment using TextBlob
def analyze_sentiment(review):
    analysis = TextBlob(review)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"

# Function to search for reviews related to a drug
def search_reviews(drug_name):
    drug_reviews = df[df['drugName'].str.contains(drug_name, case=False, na=False)]
    return f"Overall Sentiment for {drug_name}: {analyze_sentiment(' '.join(drug_reviews['review'].tolist()))}" if not drug_reviews.empty else f"No reviews found for {drug_name}."

# Function to get top drugs
def get_top_drugs(n):
    top_drugs = df["drugName"].value_counts().nlargest(n)
    return top_drugs.reset_index().rename(columns={"index": "Drug Name", "drugName": "Review Count"}) if not top_drugs.empty else "No drugs found."

# Function to get drug information
def get_drug_info(drug_name):
    drug_data = df[df["drugName"].str.lower() == drug_name.lower()]
    if drug_data.empty:
        return f"Information for {drug_name} not found."
    return f"**Rating:** {round(drug_data['rating'].mean(), 2)}\n\n**Conditions:** {', '.join(drug_data['condition'].dropna().unique())}\n\n**Summary:** {' '.join(drug_data['review'].head(5).tolist())[:300]}"

# Streamlit UI
st.title("💊 Drug Review Analysis")

# Select task
task = st.selectbox(
    "Choose an option:",
    ["Enter Review & Analyze", "Estimate Drug Rating", "Search Drug Reviews", "Top Drugs", "Know About Drug"]
)

if task == "Enter Review & Analyze":
    review_input = st.text_area("Enter a drug review:")
    if st.button("Analyze"):
        if review_input.strip():
            st.markdown(f"**Sentiment:** {classify_review(review_input)}")
        else:
            st.warning("⚠️ Please enter a review.")

elif task in ["Estimate Drug Rating", "Search Drug Reviews", "Know About Drug"]:
    drug_name = st.selectbox("Select a Drug Name:", drug_names)
    
    if task == "Estimate Drug Rating" and st.button("Estimate Rating"):
        avg_rating = df[df["drugName"].str.lower() == drug_name.lower()]["rating"].mean()
        st.success(f"**Estimated Drug Rating:** {round(avg_rating, 2)}" if not np.isnan(avg_rating) else "Rating not found.")

    elif task == "Search Drug Reviews" and st.button("Search Reviews"):
        st.markdown(search_reviews(drug_name))

    elif task == "Know About Drug" and st.button("Get Info"):
        st.markdown(get_drug_info(drug_name))

elif task == "Top Drugs":
    top_n = st.number_input("Enter the number of top drugs:", min_value=1, max_value=50, value=10, step=1)
    if st.button("Show Top Drugs"):
        st.table(get_top_drugs(top_n))
