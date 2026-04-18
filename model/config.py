import os

# Project root based on the path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths - Strategy II (ciplab dataset)
# The download script extracts into data/ciplab_dataset
DATA_DIR = os.path.join(BASE_DIR, 'data', 'ciplab_dataset')
# Note: CIPLAB dataset structure is usually 'real_and_fake_face'
RAW_DATA_PATH = os.path.join(DATA_DIR, 'real_and_fake_face')

# Processed data paths (for split sets)
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_DIR = os.path.join(PROCESSED_DIR, 'train')
VAL_DIR = os.path.join(PROCESSED_DIR, 'valid')
TEST_DIR = os.path.join(PROCESSED_DIR, 'test')

# Model output path
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'model', 'saved_models')
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_deepfake_detector.keras')

# Hyperparameters for EfficientNetB0 & smaller dataset
IMG_SIZE = (224, 224) # EfficientNetB0 standard
BATCH_SIZE = 16       # Smaller batch for smaller data/fine-tuning
EPOCHS_PHASE_1 = 15   # Initial freezing
EPOCHS_PHASE_2 = 15   # Deeper fine-tuning
LEARNING_RATE_1 = 1e-4 # Conservative LR for stability
LEARNING_RATE_2 = 5e-6 # Very low LR for fine-tuning

# Regularization
DROPOUT_RATE = 0.4
L2_REG = 1e-5

# Create save dir if not exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
