import requests
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import datetime
import torch
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from statsmodels.tsa.stattools import grangercausalitytests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import shap
import numpy as np
import sqlite3
import plotly.express as px
from newspaper import Article
from langdetect import detect
from googletrans import Translator
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from collections import Counter
import tweepy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Ensure required libraries are available
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# API Key
API_KEY = "31aea693ae334c9da178a66230a0fc45"
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Database setup
def setup_database():
    conn = sqlite3.connect("financial_sentiment.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS SentimentAnalysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            summary TEXT,
            sentiment TEXT,
            investment_score REAL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn, cursor

conn, cursor = setup_database()

# Fetch financial news
def fetch_financial_news(query="stock market", page_size=100):
    params = {
        "q": query,
        "apiKey": API_KEY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size
    }
    response = requests.get(NEWS_API_URL, params=params)
    data = response.json()
    return data["articles"] if "articles" in data else []

# Fetch stock prices
def fetch_stock_prices(ticker="AAPL", start_date="2023-01-01", end_date=str(datetime.date.today())):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Translate Non-English Text
def translate_text(text):
    try:
        if detect(text) != 'en':
            translator = Translator()
            return translator.translate(text, dest='en').text
        return text
    except:
        return text

# Summarize News Articles
def summarize_text(text):
    nltk.download('punkt')
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 2)
    return " ".join([str(sentence) for sentence in summary])

# Advanced Sentiment Analysis using BERT
sentiment_pipeline = pipeline("sentiment-analysis")

def get_sentiment(text):
    analysis = sentiment_pipeline(text)
    return analysis[0]['label']

# Investment Score Calculation
def calculate_investment_score(sentiment):
    if sentiment == "POSITIVE":
        return np.random.uniform(70, 100)
    elif sentiment == "NEGATIVE":
        return np.random.uniform(0, 40)
    else:
        return np.random.uniform(40, 70)

# Fetch and process news articles
news_articles = fetch_financial_news()
data = []

for article in news_articles:
    title = article['title']
    description = article['description'] if article['description'] else ""
    title = translate_text(title)
    description = translate_text(description)
    summary = summarize_text(description)
    combined_text = preprocess_text(f"{title} {description}")
    sentiment = get_sentiment(combined_text)
    investment_score = calculate_investment_score(sentiment)
    cursor.execute("INSERT INTO SentimentAnalysis (title, description, summary, sentiment, investment_score) VALUES (?, ?, ?, ?, ?)", (title, description, summary, sentiment, investment_score))
    data.append([title, description, summary, sentiment, investment_score])

conn.commit()
df = pd.DataFrame(data, columns=['Title', 'Description', 'Summary', 'Sentiment', 'Investment Score'])

# Streamlit Dashboard
st.title("Financial News Sentiment Analysis")
search_term = st.text_input("Search", "")
filtered_df = df[df['Title'].str.contains(search_term, case=False, na=False)]
st.dataframe(filtered_df)

st.sidebar.header("Visualization Options")
graph_option = st.sidebar.selectbox("Select Graph Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Word Cloud", "Box Plot", "Heatmap"])

if graph_option == "Bar Chart":
    st.plotly_chart(px.bar(filtered_df, x='Sentiment', y='Investment Score', title='Sentiment vs Investment Score'))
elif graph_option == "Line Chart":
    st.line_chart(filtered_df['Investment Score'])
elif graph_option == "Scatter Plot":
    st.plotly_chart(px.scatter(filtered_df, x='Sentiment', y='Investment Score', title='Sentiment vs Investment Score'))
elif graph_option == "Pie Chart":
    st.plotly_chart(px.pie(filtered_df, names='Sentiment', title='Sentiment Distribution'))
elif graph_option == "Histogram":
    st.plotly_chart(px.histogram(filtered_df, x='Investment Score', title='Investment Score Distribution'))
elif graph_option == "Word Cloud":
    wordcloud = WordCloud().generate(" ".join(filtered_df['Title']))
    st.image(wordcloud.to_array())
elif graph_option == "Box Plot":
    st.plotly_chart(px.box(filtered_df, x='Sentiment', y='Investment Score', title='Sentiment vs Investment Score'))
elif graph_option == "Heatmap":
    st.write(sns.heatmap(df.corr(), annot=True))
    st.pyplot()

# Close database connection
conn.close()
