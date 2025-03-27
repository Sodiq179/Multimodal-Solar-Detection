"""
Configuration file for the solar panel and boiler counting project.
"""

# Paths
DATA_DIR = "data/"
MODEL_SAVE_DIR = "outputs/saved_models/"
IMAGE_DIR = "data/images/"

# Training parameters
NUM_EPOCHS = 50
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_FOLDS = 5
LEARNING_RATE = 1e-4

# Image parameters
IMG_SIZE = 512

# Random seed
SEED = 42