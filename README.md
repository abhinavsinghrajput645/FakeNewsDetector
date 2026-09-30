# 📰 Fake News Detection System

A simple, beginner-friendly, and responsive Machine Learning web application to detect whether a news article is **Real** or **Fake**. Built with **Python**, **Streamlit**, and **Scikit-Learn**.

---

## 🌟 Features

- **Instant Prediction**: Classifies news articles into **Real News** or **Fake News** with confidence percentages.
- **Explainable Results**: Highlights the specific keywords in the article that influenced the prediction (words pointing to Real vs. Fake).
- **Quick Test Buttons**: One-click buttons to immediately load sample Real and Fake articles.
- **Web Link Scraper**: Option to paste any public article URL to automatically extract and analyze its text.
- **Responsive Design**: Clean and modern interface that looks great on mobile phones, tablets, and desktop screens.
- **Fact-Check Verification**: Direct links to verify claims on Google Fact Check and Snopes.
- **Viva & Interview Ready**: Includes simple explanations of how NLP (TF-IDF) and Logistic Regression classify text.

---

## 🚀 How to Run Locally

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Start the Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧠 How the Machine Learning Model Works

1. **Text Preprocessing**: The text is cleaned and converted to lowercase.
2. **TF-IDF Vectorizer**: Converts words into numerical feature vectors based on their frequency and significance.
3. **Logistic Regression Classifier**: Computes the probability that the text belongs to:
   - **Class 1**: Real News
   - **Class 0**: Fake News

---

## 📁 Project Structure

```
FakeNewsDetection/
│
├── app.py                 # Main Streamlit web application (responsive & beginner-friendly)
├── fake_news_model.pkl    # Trained Logistic Regression model
├── tfidf_vectorizer.pkl   # Trained TF-IDF vectorizer
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 📜 License
MIT License. Built for learning, education, and digital literacy.
