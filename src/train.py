from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle
from .config import MODELS_DIR, BATCH_SIZE, EPOCHS

class Trainer:
    def split_data(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_ml_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def evaluate_ml_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        return accuracy_score(y_test, y_pred)

    def train_lstm(self, model, X_train, y_train, X_test, y_test):
        filepath = os.path.join(MODELS_DIR, 'lstm_model.h5')
        checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor='val_accuracy', mode='max')
        early_stop = EarlyStopping(monitor='val_loss', patience=2)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[checkpoint, early_stop]
        )
        return history, model
    
    def save_ml_model(self, model, filename):
        path = os.path.join(MODELS_DIR, filename)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
            print(f"Saved ML model: {path}")
