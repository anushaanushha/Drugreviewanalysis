import google.generativeai as genai

# Set your API key
genai.configure(api_key="AIzaSyBKcyEBRGd-ld2LRFl_9z2grf1f5ACBU-4")

# Initialize the model
model = genai.GenerativeModel("gemini-pro")

# Function to chat with Gemini
def chat_with_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text



