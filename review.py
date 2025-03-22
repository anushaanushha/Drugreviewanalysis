import pickle
import re
import string
import torch
from transformers import BertTokenizer, BertModel

# ✅ Load the saved model, tokenizer, and condition mapping
with open("bert_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("bert_tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

with open("condition_mapping.pkl", "rb") as mapping_file:
    condition_mapping = pickle.load(mapping_file)

# ✅ Load the BERT model
BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
bert_model = BertModel.from_pretrained(BERT_MODEL)

# ✅ Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub("\\d+", "", text)  # Remove numbers
    return text

# ✅ Function to generate BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Average across tokens
    return embeddings

# ✅ Function to predict condition
def predict_condition(user_input):
    user_input_clean = preprocess_text(user_input)
    user_input_embedding = get_bert_embedding(user_input_clean).reshape(1, -1)  # Reshape for model
    predicted_id = model.predict(user_input_embedding)[0]
    return condition_mapping[predicted_id]
