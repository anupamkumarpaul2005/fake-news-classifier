from fastapi import FastAPI
import uvicorn
import pickle
import numpy as np
import tensorflow as tf
import re
import requests
from keras.preprocessing.sequence import pad_sequences



# Load model & tokenizer
model = tf.keras.models.load_model("../models/lstm_model.h5")
with open("../models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define FastAPI app
app = FastAPI()

def clean_text(text):
    text = re.sub(r"\W", " ", text)  # Remove special chars
    text = re.sub(r"\d", " ", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text.lower()

# Prediction function
def predict_fake_news(title: str, author: str, text: str):
    content = f"{title} {author} {text}"
    content = clean_text(content)
    sequence = tokenizer.texts_to_sequences([content])
    padded_seq = pad_sequences(sequence, maxlen=500, padding="post", truncating="post")
    prediction = model.predict(padded_seq)[0][0]
    label = "Fake" if prediction > 0.5 else "Real"
    return {"prediction": label, "confidence": float(prediction)}

# Google Fact Check API Key (Replace with your key)
FACT_CHECK_API_KEY = "YOUR_GOOGLE_API_KEY"

# Function to check facts using Google Fact Check API
def fact_check(query):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={FACT_CHECK_API_KEY}"
    response = requests.get(url).json()
    
    if "claims" in response and response["claims"]:
        first_claim = response["claims"][0]["claimReview"][0]
        return {
            "fact_check": first_claim["textualRating"],
            "verified_source": first_claim["publisher"]["name"]
        }
    return None

# API endpoint
@app.post("/predict")
def predict(title: str, author: str, text: str):
    return predict_fake_news(title, author, text)

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
