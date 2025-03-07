Sentimental_Analysis_ of_ News
Financial News Sentiment Analysis & Investment Scoring
Overview
This project is a comprehensive Financial News Sentiment Analysis & Investment Scoring System that fetches financial news articles, analyzes sentiment using NLP and machine learning, and provides an investment score based on sentiment. It includes a Streamlit-based interactive dashboard for visualization and insights.

Features
Real-time Financial News Retrieval: Uses the NewsAPI to fetch financial news articles.
Stock Market Data Integration: Retrieves historical stock prices using Yahoo Finance.
Text Preprocessing: Cleans and processes text using NLTK, regex, and stopword removal.
Language Detection & Translation: Translates non-English articles to English using Google Translate.
News Summarization: Uses LSA Summarizer to extract key insights from news articles.
Sentiment Analysis: Implements BERT, TextBlob, and VADER for sentiment classification.
Investment Score Calculation: Assigns an investment score based on sentiment.
Database Storage: Saves sentiment analysis results in a SQLite database.
Machine Learning Models: Implements Random Forest, SVM, Naive Bayes, and LDA for sentiment classification and topic modeling.
Data Visualization: Uses Seaborn, Matplotlib, and Plotly for insights, including bar charts, word clouds, and heatmaps.
Streamlit Dashboard: An interactive UI for financial analysts and investors.
Technologies Used
Python (Pandas, NumPy, Scikit-learn, Transformers, Matplotlib, Seaborn, Plotly, NLTK, TextBlob, VADER)
APIs (NewsAPI, Yahoo Finance, Google Translate)
Database (SQLite)
Machine Learning (BERT, Random Forest, Naive Bayes, Logistic Regression)
Streamlit (Dashboard for visualization)
How to Run
Clone the repository:
bash
Copy
Edit
git clone https://github.com/your-username/financial-news-sentiment.git
cd financial-news-sentiment
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit dashboard:
bash
Copy
Edit
streamlit run app.py
Usage
Input a search query to filter financial news.
View sentiment analysis and investment scores.
Explore stock price trends and financial insights.
Future Enhancements
Integrate deep learning models for sentiment classification.
Expand stock market prediction models.
Improve real-time news fetching and analysis.
