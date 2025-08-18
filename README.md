chintu is a Virtual Human Resource (HR) Chatbot built using PyTorch, Flask, and NLP. It can handle HR-related queries by classifying user input into predefined intents and responding intelligently.

It comes with:

🖥️ Terminal Chat (basic CLI version)

🌐 Web UI + Flask API (interactive chatbot interface)

🚀 Features

NLP preprocessing: tokenization + bag of words

Deep learning model built with PyTorch

REST API endpoint (/predict) to interact with chatbot

Simple web UI (base.html) for user interaction

Cross-Origin Resource Sharing (CORS) enabled for frontend-backend communication

Confidence thresholding (fallback if <75%)

📂 Project Structure
project/
│── model.py             # NeuralNet class (PyTorch model)
│── nltk_utils.py        # Tokenization & Bag of Words helpers
│── intents.json         # Intents and responses dataset
│── train.py             # Training script (produces data.pth)
│── chat.py              # Chat logic (getResponse function)
│── app.py               # Flask API + UI integration
│── templates/
│   └── api/
│       └── base.html    # Web UI for chatbot
│── static/              # (Optional: CSS/JS for UI styling)
│── data.pth             # Trained model state
│── requirements.txt
│── README.md

⚙️ Installation & Setup
1. Clone Repository
git clone https://github.com/yourusername/hr-chatbot.git
cd hr-chatbot

2. Install Dependencies
pip install flask flask-cors torch nltk

3. Train the Model (if not already trained)
python train.py

4. Run Flask App
python app.py


The app will start at:
👉 http://127.0.0.1:5000

📝 Usage
🌐 Web UI

Open http://127.0.0.1:5000

Type your query into the chatbot UI.

Adam will reply instantly using the model’s predictions.

📡 API Endpoint

You can also interact programmatically:

Request:

curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"message": "Hello"}'


Response:

{
  "answer": "Hello! How can I assist you today?"
}

📊 Example Intents
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey"],
      "responses": ["Hello! How can I assist you today?", "Hi there! What HR query can I help with?"]
    },
    {
      "tag": "leave_policy",
      "patterns": ["What is the leave policy?", "How many leaves do I get?"],
      "responses": ["You are entitled to 20 paid leaves annually."]
    }
  ]
}

🛠️ Tech Stack

Python

PyTorch (Deep Learning)

NLTK (Tokenization & Bag of Words)

Flask (Web Framework)

Flask-CORS (API communication)

HTML/CSS/JS (UI)
![WhatsApp Image 2025-08-18 at 22 55 53_5a25ea4a](https://github.com/user-attachments/assets/6b7a6f57-8451-40a6-b357-340f17303521)

