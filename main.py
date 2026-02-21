import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

import config
from dataset import ShapeDataset, get_transforms
from model import get_model, freeze_backbone, unfreeze_all
from train import train_one_epoch
from utils import inference, set_seed, save_checkpoint
from losses import LabelSmoothingCrossEntropy


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------
# Setup
# ----------------------------
set_seed(config.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

# ----------------------------
# Dataset
# ----------------------------
dataset = ShapeDataset(transform=get_transforms())
loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

# ----------------------------
# Model (Transfer Learning)
# ----------------------------
model = get_model(config.NUM_CLASSES)
freeze_backbone(model)
model.to(device)

print(f"Trainable parameters (head only): {count_trainable_params(model)}")

# ----------------------------
# Loss + Optimizer
# ----------------------------

# Label Smoothing Loss (prevents overconfidence)
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config.LR_HEAD
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=2,
    factor=0.1
)

# ----------------------------
# Phase 1: Train Head Only
# ----------------------------
print("\nTraining classifier head...\n")

for epoch in trange(config.EPOCHS, desc="Head Training"):
    train_loss = train_one_epoch(
        model, loader, optimizer, criterion, device, epoch=epoch+1
    )

    scheduler.step(train_loss)

    print(
        f"Epoch {epoch+1}/{config.EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    save_checkpoint(
        model, optimizer, scheduler, epoch, config.CHECKPOINT_PATH
    )

# ----------------------------
# Phase 2: Fine-Tuning (Optional)
# ----------------------------
print("\nUnfreezing entire model for fine-tuning...\n")

unfreeze_all(model)

print(f"Trainable parameters (full model): {count_trainable_params(model)}")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LR_FINE_TUNE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=2,
    factor=0.1
)

for epoch in trange(config.EPOCHS, desc="Fine-Tuning"):
    train_loss = train_one_epoch(
        model, loader, optimizer, criterion, device, epoch=epoch+1
    )

    scheduler.step(train_loss)

    print(
        f"[FT] Epoch {epoch+1}/{config.EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

print("\nTraining complete.\n")

# Inference


preds, labels = inference(model, loader, device)
accuracy = (preds == labels).sum().item() / len(labels)
print(f"Inference Accuracy: {accuracy:.4f}")