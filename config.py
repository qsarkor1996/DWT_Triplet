import os

# Configuration for paths and parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'myData.mat')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed_data')
DWT_DATA_DIR = os.path.join(BASE_DIR, 'data', 'dwt_data')
TRIPLETS_DIR = os.path.join(BASE_DIR, 'data', 'triplets')
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'data', 'visualizations')

# Hyperparameters
EMBEDDING_DIM = 128
NUM_CLASSES = 11
MARGIN = 1.0
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS_TRIPLET = 20
NUM_EPOCHS_CLASSIFICATION = 20
SNR_DB = 0  # SNR level for AWGN
DWT_WAVELET = 'db1'
DWT_LEVEL = 3
NUM_TRIPLETS = 10000