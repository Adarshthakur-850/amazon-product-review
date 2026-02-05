from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .config import MAX_WORDS, MAX_SEQ_LENGTH
import pickle
import os

class FeatureEngineer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")

    def get_tfidf_features(self, X_train, X_test):
        print("Vectorizing text (TF-IDF)...")
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        X_test_tfidf = self.tfidf.transform(X_test)
        return X_train_tfidf, X_test_tfidf

    def get_sequence_features(self, X_train, X_test):
        print("Tokenizing and Padding sequences for DL...")
        self.tokenizer.fit_on_texts(X_train)
        
        train_seq = self.tokenizer.texts_to_sequences(X_train)
        test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        X_train_pad = pad_sequences(train_seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
        X_test_pad = pad_sequences(test_seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
        
        return X_train_pad, X_test_pad, len(self.tokenizer.word_index) + 1
