import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'model/spam_model.pkl'))
tfidf = joblib.load(os.path.join(BASE_DIR, 'model/tfidf_vectorizer.pkl'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

st.title("📧 Email/SMS Spam Detector")
st.write("Enter a message below to check if it's Spam or Ham (legitimate).")

message = st.text_area("Enter your message here:")

if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(message)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        if prediction == 1:
            st.error(f"🚨 This looks like SPAM! (Confidence: {probability[1]*100:.2f}%)")
        else:
            st.success(f"✅ This looks like HAM (legitimate). (Confidence: {probability[0]*100:.2f}%)")