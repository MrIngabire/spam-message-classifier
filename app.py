import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (required for the cloud environment later)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Load our saved AI tools
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 2. Recreate our cleaning function so the app processes text exactly like the training data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# 3. Build the Streamlit Web Interface
st.title("📱 SMS Spam Detector")
st.write("Enter a text message below to check if it is Spam or Legitimate (Ham).")

# Create a text input box for the user
user_input = st.text_area("Message:", "")

# Create a button to trigger the analysis
if st.button("Analyze Message"):
    if user_input:
        # Clean the input
        cleaned_input = clean_text(user_input)
        
        # Convert words to math
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # Make the prediction
        prediction = model.predict(vectorized_input)[0]
        
        # Display the result with some visual flair
        if prediction == 'spam':
            st.error("🚨 Warning: This message looks like SPAM!")
        else:
            st.success("✅ This message appears to be SAFE (Ham).")
    else:
        st.warning("Please enter a message to analyze.")