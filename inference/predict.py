# predict.py
"""
Inference script for predicting solar panel and boiler counts using ensemble models.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from model.efficientnetv2_meta import EfficientNetV2Meta
from data.dataset import SolarPanelDataset, get_valid_transforms
from config import *


def load_model(weights_path, device):
    """Load a trained model from disk."""
    model = EfficientNetV2Meta()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_inference(df, image_dir):
    """
    Perform inference using ensemble of trained fold models.

    Args:
        df (pd.DataFrame): Test dataframe.
        image_dir (str): Directory with test images.

    Returns:
        np.ndarray: Averaged predictions of shape (N, 2)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = SolarPanelDataset(df, image_dir, transforms=get_valid_transforms(IMG_SIZE))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    predictions = []

    for fold in range(NUM_FOLDS):
        weight_path = os.path.join(MODEL_SAVE_DIR, f"best_fold{fold+1}.pth")
        model = load_model(weight_path, device)

        fold_preds = []
        with torch.no_grad():
            for images, metadata, _ in test_loader:
                images = images.to(device)
                metadata = metadata.to(device)
                outputs = model(images, metadata).cpu().numpy()
                fold_preds.append(outputs)

        fold_preds = np.vstack(fold_preds)
        predictions.append(fold_preds)

    final_preds = np.mean(predictions, axis=0)
    return final_preds


if __name__ == "__main__":
    # Example usage
    test_csv_path = "data/test.csv"
    test_image_dir = IMAGE_DIR

    test_df = pd.read_csv(test_csv_path)
    preds = run_inference(test_df, test_image_dir)

    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'boil_nbr': preds[:, 0],
        'pan_nbr': preds[:, 1],
    })

    submission.to_csv("submission.csv", index=False)
    print("Submission file saved as submission.csv")
