from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from .config import EMBEDDING_DIM, MAX_SEQ_LENGTH

class ModelFactory:
    def get_ml_model(self, model_type):
        if model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000)
        elif model_type == 'naive_bayes':
            return MultinomialNB()
        elif model_type == 'svm':
            return SVC()
        raise ValueError(f"Unknown ML model: {model_type}")

    def build_lstm_model(self, vocab_size):
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LENGTH))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dense(3, activation='softmax'))
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
