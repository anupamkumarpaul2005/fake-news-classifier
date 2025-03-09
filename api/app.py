from fastapi import FastAPI
import uvicorn
import pickle
import re
import tensorflow as tf
from pydantic import BaseModel
from keras.preprocessing.sequence import pad_sequences

# Load AI Model & Tokenizer
model = tf.keras.models.load_model("C:\DRIVE\ML\Fake News\models\lstm_model.h5")
with open("C:\DRIVE\ML\Fake News\models\\tokenizer.pkl", "rb") as f:
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
