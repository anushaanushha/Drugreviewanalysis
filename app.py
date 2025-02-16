import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import pickle
from chat import chat_with_gemini


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

# Updated list of conditions
unique_conditions = [
    "Acne", "Actinic Keratosis", "Acute Coronary Syndrome", "Acute Lymphoblastic Leukemia", "Acute Promyelocytic Leukemia",
    "Addison's Disease", "Adrenocortical Insufficiency", "Adult Human Growth Hormone Deficiency", "Agitation", "Alcohol Dependence",
    "Allergic Rhinitis", "Alopecia", "Alzheimer's Disease", "Amenorrhea", "Amyotrophic Lateral Sclerosis", "Anxiety",
    "Anxiety and Stress", "Arrhythmia", "Asthma", "Atrial Fibrillation", "Bacterial Infection", "Bacterial Skin Infection",
    "Benign Prostatic Hyperplasia", "Bipolar Disorder", "Bladder Infection", "Breast Cancer", "Bronchitis", "Cancer",
    "Chronic Fatigue Syndrome", "Chronic Myeloid Leukemia", "Chronic Pain", "COPD (Chronic Obstructive Pulmonary Disease)",
    "Constipation", "Crohn's Disease", "Cystic Fibrosis", "Depression", "Diabetes Type 1", "Diabetes Type 2", "Diabetic Neuropathy",
    "Diarrhea", "Diverticulitis", "Dry Eye Disease", "Dysmenorrhea", "Eczema", "Endometriosis", "Epilepsy", "Erectile Dysfunction",
    "Fibromyalgia", "GERD (Gastroesophageal Reflux Disease)", "Glaucoma", "Gout", "Headache", "Heart Failure", "Hepatitis B",
    "Hepatitis C", "High Blood Pressure (Hypertension)", "High Cholesterol", "HIV Infection", "Hyperthyroidism", "Hypothyroidism",
    "Insomnia", "Irritable Bowel Syndrome", "Kidney Infections", "Leukemia", "Liver Disease", "Lyme Disease", "Lupus",
    "Macular Degeneration", "Malaria", "Menopause", "Migraine", "Multiple Sclerosis", "Nausea/Vomiting", "Obesity", "Osteoarthritis",
    "Osteoporosis", "Panic Disorder", "Parkinson's Disease", "Peptic Ulcer Disease", "Pneumonia", "Postmenopausal Symptoms",
    "Post-Traumatic Stress Disorder (PTSD)", "Prostate Cancer", "Psoriasis", "Psoriatic Arthritis", "Rheumatoid Arthritis", "Rosacea",
    "Schizophrenia", "Sciatica", "Seizures", "Sexual Dysfunction", "Shingles", "Sinusitis", "Skin Cancer", "Sleep Apnea",
    "Smoking Cessation", "Stomach Ulcer", "Stroke", "Urinary Tract Infection (UTI)", "Vitamin D Deficiency"
]

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Function to classify sentiment using trained model and highlight key element
def classify_review(review):
    cleaned_review = clean_text(review)
    X_input = vectorizer.transform([cleaned_review])
    prediction = model.predict(X_input)

    # Extract feature names from the vectorizer
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Get the word importance (TF-IDF scores)
    tfidf_scores = X_input.toarray()[0]

    # Find the most important word(s) based on highest TF-IDF score
    top_word_index = np.argsort(tfidf_scores)[-1]  # Get index of max score
    top_word = feature_names[top_word_index]

    sentiments = ["negative", "neutral", "positive"]
    sentiment_result = sentiments[int(prediction[0])] if prediction.ndim == 1 else sentiments[np.argmax(prediction, axis=1)[0]]
    
    return sentiment_result, top_word

# Function to analyze sentiment using TextBlob
def analyze_sentiment(review):
    analysis = TextBlob(review)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"

# Function to search for reviews related to a drug
def search_reviews(drug_name):
    drug_reviews = df[df['drugName'].str.contains(drug_name, case=False, na=False)]
    return f"Overall Sentiment for {drug_name}: {analyze_sentiment(' '.join(drug_reviews['review'].tolist()))}" if not drug_reviews.empty else f"No reviews found for {drug_name}."

# Function to get top drugs based on condition and rating
def get_top_drugs_by_condition(condition):
    filtered_df = df[df["condition"].str.contains(condition, case=False, na=False)]
    if filtered_df.empty:
        return "No drugs found for this condition."
    
    top_drugs = (
        filtered_df.groupby("drugName")["rating"]
        .mean()
        .reset_index()
        .sort_values(by="rating", ascending=False)
    )
    return top_drugs.rename(columns={"drugName": "Drug Name", "rating": "Average Rating"})

# Function to get drug information
def get_drug_info(drug_name):
    drug_data = df[df["drugName"].str.lower() == drug_name.lower()]
    if drug_data.empty:
        return f"Information for {drug_name} not found."
    return f"**Rating:** {round(drug_data['rating'].mean(), 2)}"
def get_drug_information(drug_name):
    drug_data = df[df["drugName"].str.lower() == drug_name.lower()]
    if drug_data.empty:
        return f"Information for {drug_name} not found."
    return f"*Rating:* {round(drug_data['rating'].mean(), 2)}\n\n*Conditions:* {', '.join(drug_data['condition'].dropna().unique())}\n\n*Summary:* {' '.join(drug_data['review'].head(5).tolist())[:300]}"

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
            sentiment, key_word = classify_review(review_input)
            st.markdown(f"**Sentiment:** {sentiment}")
            st.markdown(f"**Key Element:** `{key_word}` ")
        else:
            st.warning("⚠️ Please enter a review.")

elif task == "Top Drugs":
    selected_condition = st.selectbox("Select Condition:", unique_conditions)
    if st.button("Show Top Drugs"):
        st.table(get_top_drugs_by_condition(selected_condition))

elif task in ["Estimate Drug Rating", "Search Drug Reviews", "Know About Drug"]:
    selected_drug = st.selectbox("Select Drug:", drug_names)
    if st.button("Analyze"):
        if task == "Estimate Drug Rating":
            st.markdown(get_drug_info(selected_drug))
        elif task == "Search Drug Reviews":
            st.markdown(search_reviews(selected_drug))
        elif task == "Know About Drug":
            st.markdown(get_drug_information(selected_drug))


# Update for image display (Replace `use_column_width` with `use_container_width`)
st.sidebar.image("image.jpg", use_container_width=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores chat history

# Chat history section
with st.sidebar.expander("📜 Chat History", expanded=True):
    for message in st.session_state.messages:
        role = "🧑 You" if message["role"] == "You" else "🤖 Chatbot"
        st.markdown(f"**{role}:** {message['text']}")

# Input field for user message (Avoid modifying session_state directly after defining it)
user_input = st.sidebar.text_input("Type your message here...", key="user_input")

if st.sidebar.button("Send"):
    if user_input.strip():
        # Append user input to chat history
        st.session_state.messages.append({"role": "You", "text": user_input})

        try:
            # Get response from chatbot
            response = chat_with_gemini(user_input)
        except Exception as e:
            response = "⚠️ Chatbot service is currently unavailable. Please try again later."

        # Append chatbot response to chat history
        st.session_state.messages.append({"role": "Chatbot", "text": response})

        # Reset input field (Instead of modifying session_state, we use st.experimental_rerun)
        st.rerun()
    else:
        st.sidebar.warning("⚠️ Please enter a message.")
