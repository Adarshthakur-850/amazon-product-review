import pandas as pd
from .config import DATASET_URL

class DataLoader:
    def load_data(self, limit=5000):
        print("Loading dataset...")
        df = pd.read_csv(DATASET_URL, nrows=limit)
        return df

    def clean_dataset(self, df):
        print("Cleaning dataset...")
        print(f"Columns found: {df.columns.tolist()}")
        
        # Mapping common column names
        col_map = {}
        for col in df.columns:
            if col.lower() in ['reviewtext', 'text', 'review']:
                col_map[col] = 'Text'
            elif col.lower() in ['score', 'rating', 'stars', 'positive']: # 'Positive' is in PyCaret
                col_map[col] = 'Score'
        
        if col_map:
            df = df.rename(columns=col_map)
        
        if 'Score' not in df.columns or 'Text' not in df.columns:
            raise ValueError(f"Required columns not found. Mapped: {df.columns.tolist()}")
            
        df = df[['Score', 'Text']]
             
        df = df.dropna()
        df = df.drop_duplicates()
        return df
