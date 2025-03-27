"""
Main entry point for training and evaluation.
"""

import pandas as pd
from train.train import run_training
from config import IMAGE_DIR


def main():
    print("Starting training pipeline...")
    train_csv_path = "data/train.csv"
    train_df = pd.read_csv(train_csv_path)
    run_training(train_df, IMAGE_DIR)


if __name__ == "__main__":
    main()
