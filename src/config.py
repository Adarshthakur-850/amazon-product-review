import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DATASET_URL = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv"

MAX_WORDS = 10000
MAX_SEQ_LENGTH = 100
EMBEDDING_DIM = 100
EPOCHS = 5
BATCH_SIZE = 32
