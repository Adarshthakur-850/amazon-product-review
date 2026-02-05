from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.visualization import Visualizer
from src.models import ModelFactory
from src.train import Trainer
import pandas as pd
import numpy as np

def main():
    print("=== Amazon Product Review Sentiment Analysis ===")
    
    loader = DataLoader()
    df = loader.load_data(limit=5000)
    df = loader.clean_dataset(df)
    print(f"Dataset Shape: {df.shape}")
    
    preprocessor = Preprocessor()
    df = preprocessor.process(df)
    
    viz = Visualizer()
    viz.plot_sentiment_dist(df)
    pos_text = " ".join(df[df['sentiment'] == 2]['cleaned_text'])
    if pos_text:
        viz.plot_wordcloud(pos_text, "Positive Reviews")
        
    fe = FeatureEngineer()
    X = df['cleaned_text']
    y = df['sentiment'].values
    
    trainer = Trainer()
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    X_train_tfidf, X_test_tfidf = fe.get_tfidf_features(X_train, X_test)
    model_factory = ModelFactory()
    
    print("\n--- Training Logistic Regression ---")
    lr = model_factory.get_ml_model('logistic_regression')
    lr = trainer.train_ml_model(lr, X_train_tfidf, y_train)
    acc_lr = trainer.evaluate_ml_model(lr, X_test_tfidf, y_test)
    trainer.save_ml_model(lr, 'logistic_regression.pkl')
    
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")

    print("\n--- Training LSTM ---")
    X_train_seq, X_test_seq, vocab_size = fe.get_sequence_features(X_train, X_test)
    
    lstm_model = model_factory.build_lstm_model(vocab_size)
    history, trained_lstm = trainer.train_lstm(lstm_model, X_train_seq, y_train, X_test_seq, y_test)
    
    viz.plot_training_history(history)
    print("LSTM training complete.")
    
    print("\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
