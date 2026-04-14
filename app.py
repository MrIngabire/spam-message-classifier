import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

# Download stopwords (required for the cloud environment)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Load our saved AI tools
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 2. Recreate our cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# 3. Build the Streamlit Web Interface
st.title("📱 SMS Spam Detector")
st.write("Enter a text message below to check if it is Spam or Legitimate (Ham).")

user_input = st.text_area("Message:", "")

if st.button("Analyze Message"):
    if user_input:
        # Clean and convert the input
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # Make the prediction AND get the probability
        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]
        
        # Probabilities return an array like [Ham_Prob, Spam_Prob]. We want the Spam_Prob.
        spam_probability = probabilities[1] * 100 
        
        # --- NEW: THE ANALYTICS DASHBOARD ---
        
        st.divider() # Adds a clean horizontal line
        
        # 1. The Result & Confidence Meter
        st.subheader("AI Analysis Result")
        
        if prediction == 'spam':
            st.error(f"🚨 **SPAM DETECTED**")
        else:
            st.success(f"✅ **SAFE (HAM)**")
            
        st.write(f"**Confidence Score:** The AI is {spam_probability:.1f}% sure this message is spam.")
        st.progress(int(spam_probability)) # Creates a visual progress bar
        
        # 2. The Message Breakdown (Bar Chart)
        st.subheader("🔍 Why did the AI make this decision?")
        st.write("Here are the specific words from your message that carried the most mathematical weight:")
        
        # Extract the specific words and their TF-IDF scores from this message
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = vectorized_input.toarray()[0]
        
        # Match the words to their scores
        word_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names)) if tfidf_scores[i] > 0}
        
        if word_scores:
            # Convert to a Pandas table, sort them, and grab the top 5
            df_words = pd.DataFrame(list(word_scores.items()), columns=['Word', 'Importance Score'])
            df_words = df_words.sort_values(by='Importance Score', ascending=False).head(5)
            
            # Draw a Streamlit Bar Chart
            st.bar_chart(df_words.set_index('Word'))
        else:
            st.write("*No highly significant keywords were detected in this message.*")

    else:
        st.warning("Please enter a message to analyze.")
