import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import os
import datetime
from nltk.corpus import stopwords

# --- CONFIGURATION & ASSETS ---
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

# --- MODULE 1: DATA PROCESSOR (UOK Chapter 4.6 - Functional Description) ---
class DataProcessor:
    """Handles text cleaning and validation logic."""
    @staticmethod
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        cleaned_words = [word for word in words if word not in STOP_WORDS]
        return ' '.join(cleaned_words)

    @staticmethod
    def validate_input(text):
        """Unit Testing Logic for Validation (UOK Chapter 4.10.4)"""
        if not text.strip():
            return False, "Input cannot be empty."
        if len(text.strip()) < 3:
            return False, "Message too short for meaningful analysis."
        return True, ""

# --- MODULE 2: INFERENCE ENGINE (UOK Chapter 4.6 - System Configuration) ---
class SpamEngine:
    """Handles the AI brain and mathematical weight calculation."""
    def __init__(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text):
        cleaned = DataProcessor.clean_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        prob = self.model.predict_proba(vectorized)[0][1] * 100
        
        # Extract word weights for the 'Why' analysis
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = vectorized.toarray()[0]
        word_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names)) if tfidf_scores[i] > 0}
        
        return prediction, prob, word_scores

# --- MODULE 3: SYSTEM LOGGING (UOK Chapter 4.2 - Data Presentation) ---
def log_system_activity(message, result, confidence):
    """Logs system results for subsequent Chapter 4 interpretation."""
    log_file = 'system_usage_logs.csv'
    log_entry = pd.DataFrame([{
        'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Input_Snippet': message[:30] + "...",
        'Result': result.upper(),
        'Confidence': f"{confidence:.1f}%"
    }])
    log_entry.to_csv(log_file, mode='a', index=False, header=not os.path.exists(log_file))

# --- MODULE 4: USER INTERFACE (UOK Chapter 4.8 - Front-End Architecture) ---
def main():
    st.set_page_config(page_title="UOK AI Research Project", layout="centered")
    st.title("📱 Intelligent SMS Spam Filtering System")
    st.markdown("---")
    
    # Initialize Engine
    try:
        engine = SpamEngine('spam_model.pkl', 'vectorizer.pkl')
    except FileNotFoundError:
        st.error("System Error: AI Model files missing. Please upload pkl files.")
        return

    # User Input Area
    user_input = st.text_area("Enter Message for Analysis:", height=100)

    if st.button("Run AI Diagnostic"):
        # 1. Validation (Unit Testing)
        is_valid, error_msg = DataProcessor.validate_input(user_input)
        
        if is_valid:
            # 2. Execution
            prediction, confidence, word_weights = engine.predict(user_input)
            
            # 3. Logging (For Chapter 4 Data Analysis)
            log_system_activity(user_input, prediction, confidence)
            
            # 4. Display Results
            st.subheader("Analysis Output")
            if prediction == 'spam':
                st.error("🚨 CLASSIFICATION: SPAM")
            else:
                st.success("✅ CLASSIFICATION: SAFE (HAM)")
            
            st.metric("AI Confidence Score", f"{confidence:.1f}%")
            st.progress(int(confidence))

            # 5. Visual Dashboard (Explanatory AI)
            if word_weights:
                st.subheader("🔍 Mathematical Feature Weighting")
                df_viz = pd.DataFrame(list(word_weights.items()), columns=['Keyword', 'Weight'])
                df_viz = df_viz.sort_values(by='Weight', ascending=False).head(5)
                st.bar_chart(df_viz.set_index('Keyword'))
        else:
            st.warning(error_msg)

if __name__ == "__main__":
    main()
