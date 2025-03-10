from fastapi import FastAPI
import uvicorn
import pickle
import re
import tensorflow as tf
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from keras.preprocessing.sequence import pad_sequences # type: ignore
import os
import time


REPO_ID = "Yoppsoic/fake-news-lstm-model"
# Load AI Model & Tokenizer
'''
MODEL_PATH = os.path.join(os.getcwd(), "models/lstm_model.h5")
TOKENIZER_PATH = os.path.join(os.getcwd(), "models/tokenizer.pkl")
'''

os.makedirs("/app/models", exist_ok=True)

# Loading the models from hugging face which we uploaded using "../hf_upload.py"
print("Downloading model from Hugging Face...")
start_time = time.time()
MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename="lstm_model.h5", cache_dir="/app/models")
TOKENIZER_PATH = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.pkl", cache_dir="/app/models")
print(f"✅ Model downloaded in {time.time() - start_time:.2f} seconds!")
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Base Model of the Input
class NewsInput(BaseModel):
    title: str
    text: str
    author: str = "Unknown"

# Initialize FastAPI app
app = FastAPI()

def clean_text(text):
    text = re.sub(r"\W", " ", text)  # Remove special chars
    text = re.sub(r"\d", " ", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text.lower()

# Function to predict Fake/Real using AI Model
def predict_fake_news(title: str, text: str, author: str = "Unknown"):
    content = f"{title} {author} {text}"
    content = clean_text(content)
    sequence = tokenizer.texts_to_sequences([content])
    padded_seq = pad_sequences(sequence, maxlen=500, padding="post", truncating="post")
    prediction = model.predict(padded_seq)[0][0]
    label = "Fake" if prediction > 0.5 else "Real"
    confidence = max(float(prediction), 1 - float(prediction))
    return {"prediction": label, "confidence": confidence}

# API Endpoint
@app.post("/predict")
def predict(news: NewsInput):
    ai_result = predict_fake_news(news.title, news.text, news.author)
    return {
        "prediction": ai_result["prediction"],
        "confidence": ai_result["confidence"]
    }

# Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
