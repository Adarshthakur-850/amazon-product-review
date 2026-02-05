import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return " ".join(words)

    def map_sentiment(self, score):
        # If score is already 0, 1, 2 (mapped) leave it? 
        # But here we assume input is Raw.
        # If binary 0/1: 1=Pos(2), 0=Neg(0)
        # If 1-5: >3 Pos(2), <3 Neg(0), 3 Neu(1)
        return score # Placeholder, logic moved to process
        
    def process(self, df):
        print("Preprocessing text...")
        df['cleaned_text'] = df['Text'].apply(self.preprocess_text)
        
        # Determine unique values to guess scale
        unique_scores = df['Score'].unique()
        max_score = df['Score'].max()
        
        if max_score <= 1:
            # Assume Binary 0/1
            print("Detected Binary Sentiment (0/1)")
            df['sentiment'] = df['Score'].apply(lambda x: 2 if x == 1 else 0)
        else:
            # Assume 1-5 Scale
            print("Detected 1-5 Sentiment Scale")
            def map_scale(s):
                if s > 3: return 2
                elif s < 3: return 0
                else: return 1
            df['sentiment'] = df['Score'].apply(map_scale)
            
        return df
