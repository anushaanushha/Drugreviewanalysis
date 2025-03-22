import google.generativeai as genai

# Set your API key
genai.configure(api_key="AIzaSyCXjsxhKYs3lllhjg_zI-LvQi5ChOcO3x0")

# Initialize the model
model = genai.GenerativeModel("gemini-1.5-pro")

# Function to chat with Gemini
def chat_with_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text