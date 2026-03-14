from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rf = pickle.load(open('spam_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

class EmailInput(BaseModel):
    text: str

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

@app.post('/predict')
def predict(email: EmailInput):
    cleaned = clean_text(email.text)
    features = tfidf.transform([cleaned]).toarray()
    prediction = rf.predict(features)[0]
    confidence = rf.predict_proba(features)[0][prediction]
    return {
        'prediction': 'spam ' if prediction == 1 else 'ham ',
        'confidence': round(float(confidence) * 100, 2)
    }

@app.get('/')
def home():
    return FileResponse('index.html')
