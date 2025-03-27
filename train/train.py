# train.py
"""
Training script for solar panel and boiler counting using EfficientNetV2Meta.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import numpy as np

from model.efficientnetv2_meta import EfficientNetV2Meta
from data.dataset import SolarPanelDataset, get_train_transforms, get_valid_transforms
from config import *


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    for images, metadata, targets in tqdm(loader, desc="Training"):
        images, metadata, targets = images.to(device), metadata.to(device), targets.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images, metadata)
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """Validate model on a fold."""
    model.eval()
    predictions, targets_list = [], []

    with torch.no_grad():
        for images, metadata, targets in tqdm(loader, desc="Validating"):
            images, metadata = images.to(device), metadata.to(device)
            outputs = model(images, metadata).cpu().numpy()
            targets = targets.numpy()
            predictions.append(outputs)
            targets_list.append(targets)

    predictions = np.vstack(predictions)
    targets_list = np.vstack(targets_list)
    return mean_absolute_error(targets_list, predictions)


def run_training(df, image_dir):
    """
    Train the model using K-Fold cross-validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n----- Fold {fold + 1} / {NUM_FOLDS} -----")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_ds = SolarPanelDataset(train_df, image_dir, transforms=get_train_transforms(IMG_SIZE))
        val_ds = SolarPanelDataset(val_df, image_dir, transforms=get_valid_transforms(IMG_SIZE))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = EfficientNetV2Meta().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.HuberLoss()
        scaler = torch.cuda.amp.GradScaler()

        best_mae = float('inf')
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
            val_mae = validate(model, val_loader, device)

            print(f"Train Loss: {train_loss:.4f} | Validation MAE: {val_mae:.4f}")

            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"best_fold{fold+1}.pth"))

        fold_scores.append(best_mae)
        print(f"Best MAE for Fold {fold+1}: {best_mae:.4f}")

    print("\nOverall MAE scores across folds:", fold_scores)
    print("Mean MAE:", np.mean(fold_scores))
