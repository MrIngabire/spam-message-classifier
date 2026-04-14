import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import os
import datetime
from nltk.corpus import stopwords
import plotly.express as px

# --- CONFIGURATION & ASSETS ---
# Download stopwords (required for the cloud environment)
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

# --- MODULE 1: DATA PROCESSOR (UOK Functional Module) ---
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
        """Unit Testing Logic for Validation."""
        if not text.strip():
            return False, "Input cannot be empty."
        if len(text.strip()) < 3:
            return False, "Message too short for meaningful analysis."
        return True, ""

# --- MODULE 2: INFERENCE ENGINE (UOK System Configuration) ---
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
        
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = vectorized.toarray()[0]
        word_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names)) if tfidf_scores[i] > 0}
        
        return prediction, prob, word_scores

# --- MODULE 3: SYSTEM LOGGING (UOK Data Presentation) ---
def log_system_activity(message, result, confidence):
    """Logs system results for Chapter 4 interpretation."""
    log_file = 'system_usage_logs.csv'
    log_entry = pd.DataFrame([{
        'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Input_Snippet': message[:30] + "...",
        'Result': result.upper(),
        'Confidence': f"{confidence:.1f}%"
    }])
    log_entry.to_csv(log_file, mode='a', index=False, header=not os.path.exists(log_file))

# --- MODULE 4: USER INTERFACE (UOK Front-End Architecture) ---
def main():
    # UOK Formatting: Professional Layout
    st.set_page_config(page_title="Intelligent Spam Diagnostic", layout="wide")

    # Custom CSS for the "Impress Mode" Dark Theme
    st.markdown("""
        <style>
        .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
        [data-testid="stMetricValue"] { color: #58a6ff; }
        </style>
        """, unsafe_allow_html=True)

    st.title("📱 Intelligent SMS Spam Filtering System")
    st.caption("Final Year Research Project - AI-Based Task Streamlining Sub-module")
    
    # Initialization
    try:
        engine = SpamEngine('spam_model.pkl', 'vectorizer.pkl')
    except Exception:
        st.error("⚠️ SYSTEM ERROR: AI Model assets not found. Check deployment folder.")
        return

    # TOP ROW: System Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("System Status", "ACTIVE", "Healthy")
    m2.metric("Inference Engine", "NB-V1.4", "Stable")
    
    # Load logs to show real metrics if file exists
    if os.path.exists('system_usage_logs.csv'):
        log_df = pd.read_csv('system_usage_logs.csv')
        m3.metric("Scanned Items", len(log_df), f"+{len(log_df)} total")
        spam_count = len(log_df[log_df['Result'] == 'SPAM'])
        m4.metric("Spam Detected", spam_count)
    else:
        m3.metric("Scanned Items", "0")
        m4.metric("Spam Detected", "0")

    st.divider()

    # MAIN ROW: Split Dashboard
    col_input, col_viz = st.columns([1, 1])

    with col_input:
        st.subheader("📥 Live Diagnostic Tool")
        user_input = st.text_area("Paste SMS content here:", height=150, placeholder="Type message...")
        
        if st.button("RUN AI CLASSIFICATION", use_container_width=True):
            is_valid, error_msg = DataProcessor.validate_input(user_input)
            
            if is_valid:
                prediction, confidence, word_weights = engine.predict(user_input)
                log_system_activity(user_input, prediction, confidence)
                
                # Result Box
                if prediction == 'spam':
                    st.error(f"### 🚨 RESULT: SPAM DETECTED")
                else:
                    st.success(f"### ✅ RESULT: LEGITIMATE (HAM)")
                
                st.write(f"**Confidence Level:** {confidence:.2f}%")
                st.progress(int(confidence))
            else:
                st.warning(error_msg)

    with col_viz:
        st.subheader("🔍 Algorithmic Interpretation")
        if 'word_weights' in locals() and word_weights:
            st.write("Relative weight of keywords found in the message:")
            df_viz = pd.DataFrame(list(word_weights.items()), columns=['Keyword', 'Score'])
            df_viz = df_viz.sort_values(by='Score', ascending=False).head(8)
            
            # Using Plotly for a more "Impressive" chart than standard st.bar_chart
            fig = px.bar(df_viz, x='Score', y='Keyword', orientation='h', 
                         color='Score', color_continuous_scale='Blues')
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Input a message and run analysis to view keyword weighting.")

    # BOTTOM ROW: History Table
    st.divider()
    st.subheader("📊 Historical System Logs")
    if os.path.exists('system_usage_logs.csv'):
        history_df = pd.read_csv('system_usage_logs.csv')
        st.dataframe(history_df.tail(5), use_container_width=True)
    else:
        st.write("No logs available yet. Perform a scan to generate data.")

if __name__ == "__main__":
    main()
