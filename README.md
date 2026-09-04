# Restaurant Sentiment Analysis

A Natural Language Processing and Deep Learning project that classifies restaurant reviews as positive or negative and provides the prediction through a deployed Streamlit application.

## Live Application

[Open the Restaurant Sentiment Analysis App](https://geeta170817-sentiment-analysis-app-cxzmga.streamlit.app/)

## Project Objective

The goal is to analyse restaurant review text and predict customer sentiment:

- Positive Review
- Negative Review

## Approach

1. Load restaurant review data
2. Clean and normalize text
3. Convert review text into numerical features using CountVectorizer
4. Train an Artificial Neural Network model with TensorFlow/Keras
5. Save the trained model and fitted vectorizer
6. Build a Streamlit interface for real-time predictions
7. Display sentiment and prediction probability

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- CountVectorizer
- Streamlit
- Pickle
- Regular Expressions

## Application Features

- Accepts a restaurant review from the user
- Cleans the review text
- Converts the text using the saved CountVectorizer
- Predicts sentiment using the trained ANN model
- Shows Positive or Negative result
- Displays prediction probability

## Repository Contents

- `RestaurantReview.ipynb` — model development notebook
- `Restaurant_Reviews.tsv` — review dataset
- `restaurant_sentiment.keras` — trained Keras model
- `cv.pkl` — fitted CountVectorizer
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies

## Skills Demonstrated

Python, NLP preprocessing, text vectorization, Artificial Neural Networks, TensorFlow/Keras, model persistence, Streamlit deployment, and end-to-end ML application development.