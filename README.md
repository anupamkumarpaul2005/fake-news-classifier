﻿# 📰 Fake News Detection API & UI  
🚀 **AI-powered Fake News Detection system using NLP, TensorFlow, FastAPI, and Streamlit**  


## 📌 Overview  
This project aims to **detect fake news articles** using **Deep Learning (LSTM)** to improve accuracy.  
It provides:  
✅ **FastAPI** backend for model inference.  
✅ **Streamlit** UI for user-friendly interaction.  
✅ **Dockerized** setup for easy deployment.  


## 🛠️ Features
✔ **LSTM Model** trained on Kaggle's Fake News Dataset.   
✔ **FastAPI Backend** for real-time predictions.  
✔ **Streamlit UI** for easy news verification.  
✔ **Dockerized Setup** for portability.  
✔ **Hugging Face Model Hosting** for lightweight deployment.  


## 🚀 Installation
### 1️⃣ Clone the Repository
```
$ git clone https://github.com/anupamkumarpaul2005
$ fake-news-classifier.git
$ cd fake-news-classifier
```
### 2️⃣ Setup Python Environment
```
$ python -m venv venv
$ source venv/bin/activate   # For Linux/macOS
$ venv\Scripts\activate      # For Windows

$ pip install -r requirements.txt
```
### 3️⃣ Run FastAPI Backend
```
$ cd api
$ uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
📌 API Documentation: Open http://127.0.0.1:8000/docs in your browser.
### 4️⃣ Run Streamlit UI
```
$ cd ui
$ streamlit run ui.py
```
📌 Access UI at: http://127.0.0.1:8501/


## 🐳 Docker Setup
### 1️⃣ Build & Run Docker Containers
```
$ docker-compose up --build
```
📌 **FastAPI (Backend)**: http://localhost:8000  
📌 **Streamlit (Frontend)**: http://localhost:8501
### 2️⃣ Stop Containers
```
$ docker-compose down
```


## 🧠 How It Works
1️⃣ User submits news article (title, author, text).  
2️⃣ LSTM Model predicts if it’s fake or real.


## 📊 Model Performance

| Metric | Value |
|--------|-------|
|Accuracy|99.15%|
|Precision|99.46%|
|Recall|98.87%|
|F1-Score|99.16%|

## 📂 Datasets & Model Links

📌 **Original Dataset (Kaggle)**  
[🔗 Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/)  
[🔗 Another Dataset from a Competition (Only train.csv)](https://www.kaggle.com/c/fake-news/data)

📌 **Processed Dataset (Uploaded to Kaggle)**  
[🔗 Processed Fake News Dataset](https://www.kaggle.com/datasets/anupampaul005/fake-news-dataset)  

📌 **Trained Model (Hugging Face)**  
[🔗 Fake News Detection Model](https://huggingface.co/Yoppsoic/fake-news-lstm-model/tree/main)  


## 👨‍💻 Contributors
👤 Anupam Kumar Paul  
📌 AI/ML Enthusiast  
🔗 [LinkedIn](https://www.linkedin.com/in/anupamkumarpaul/) | [GitHub](https://github.com/anupamkumarpaul2005)

## License
MIT License - see the [LICENSE](LICENSE) file for details.

---
### 💡 If you like this project, give it a ⭐ on GitHub!
---
