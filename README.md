# Amazon Product Review Sentiment Analysis

A production-quality NLP system that classifies Amazon product reviews into Positive, Negative, or Neutral sentiments using standard ML (Logistic Regression) and Deep Learning (LSTM).

## Project Structure
```
amazon product review/
├── models/             # Saved .pkl and .h5 models
├── plots/              # Generated visualizations
├── src/
│   ├── config.py       # Configuration and constants
│   ├── data_loader.py  # Data fetching and cleaning
│   ├── preprocessing.py# Text cleaning and sentiment labeling
│   ├── feature_engineering.py # TF-IDF and Tokenization
│   ├── visualization.py# Plotting functions
│   ├── models.py       # Model definitions (ML & LSTM)
│   ├── train.py        # Training and evaluation logic
├── main.py             # Entry point
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Features
- **Data Pipeline**: Loading and cleaning Amazon reviews.
- **Preprocessing**: NLTK-based cleaning (removal of stopwords, punctuation, lemmatization).
- **Features**: TF-IDF (for ML) and Word Embeddings (for LSTM).
- **Models**:
  - Logistic Regression (Baseline)
  - LSTM (Deep Learning with Embeddings)
- **Evaluation**: Accuracy, Classification Report, and Training History.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have TensorFlow installed suitable for your hardware.*

## Usage

Run the main pipeline:
```bash
python main.py
```

## Output
- **Plots**: `plots/` contains Sentiment Distribution, Word Cloud, and Training History.
- **Models**: `models/` contains `logistic_regression.pkl` and `lstm_model.h5`.
