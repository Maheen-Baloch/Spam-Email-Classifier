# 📧 Spam Email Classifier

A full stack machine learning web app that classifies emails as **spam or ham** using Natural Language Processing and a Random Forest model — served via a FastAPI backend and a clean HTML/CSS/JS frontend.

---

## 🚀 Demo

> Paste any email → click Analyse → get instant prediction with confidence score

![Spam Classifier UI](Screenshot.png)

---

## 🧠 How It Works

```
Raw Email Text
      ↓
Text Preprocessing (lowercase, remove punctuation, stopwords, stemming)
      ↓
TF-IDF Vectorization (top 3000 features)
      ↓
Random Forest Classifier
      ↓
Spam 🚨 or Ham ✅ + Confidence %
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Naive Bayes | 97.31% | 99.17% | 80.54% | 88.89% |
| Logistic Regression | 96.59% | 99.12% | 75.17% | 85.50% |
| **Random Forest** ✅ | **97.67%** | **98.43%** | **83.89%** | **90.58%** |

> Random Forest was selected as the final model based on best F1 Score.

---

## 🗂️ Project Structure

```
spam-classifier/
├── app.py               # FastAPI backend
├── model.ipynb          # Model training notebook
├── index.html           # Frontend UI
├── spam_model.pkl       # Trained Random Forest model
├── tfidf.pkl            # Fitted TF-IDF vectorizer
├── spam.csv             # Dataset (SMS Spam Collection)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

**Machine Learning**
- Python, Pandas, NumPy
- Scikit-learn (Random Forest, TF-IDF)
- NLTK (stopwords, stemming)

**Backend**
- FastAPI
- Uvicorn
- Pickle

**Frontend**
- HTML5, CSS3, JavaScript
- Fetch API

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/spam-classifier.git
cd spam-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI server
```bash
uvicorn app:app --reload
```

### 4. Open the frontend
Open `index.html` in your browser — or serve it locally:
```bash
python -m http.server 3000
```

Then visit `http://localhost:3000`

---

## 📦 Dataset

**SMS Spam Collection Dataset**
- 5,572 messages labeled spam or ham
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- Class distribution: 86.6% ham, 13.4% spam

---

## 📈 Text Preprocessing Pipeline

```python
def clean_text(text):
    text = text.lower()                          # lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # remove punctuation
    words = text.split()                          # tokenize
    words = [ps.stem(w) for w in words           # stem
             if w not in stop_words]             # remove stopwords
    return ' '.join(words)
```

---

## 📋 Requirements

```
fastapi
uvicorn
scikit-learn
nltk
pandas
numpy
```

---

## 👤 Author

Maheen Baloch

---

## 📄 License

MIT License
