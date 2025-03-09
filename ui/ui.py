import streamlit as st
import requests

# FastAPI backend URL
FASTAPI_URL = "http://fastapi:8000/predict"
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰"
)
st.title("📰 Fake News Detector")
st.write("Enter the news details below to check if it's real or fake.")

title = st.text_input("News Title", "")
author = st.text_input("Author (Optional)", "Unknown")
text = st.text_area("News Content", "")

if st.button("Check Reliability"):
    if title.strip() == "" or text.strip() == "":
        st.warning("⚠️ Please enter both the title and content!")
    else:
        response = requests.post(FASTAPI_URL, json={"title": title, "text": text, "author": author})
        if response.status_code == 200:
            result = response.json()
            st.subheader("Prediction Result")
            st.write(f"🧐 **Prediction:** {result['prediction']}")
            st.write(f"📊 **Confidence:** {result['confidence']:.2f}")
        else:
            st.error("❌ Error connecting to FastAPI!")