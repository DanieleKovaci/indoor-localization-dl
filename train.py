"""
train.py — Training script for UJIIndoorLoc building/floor classification.

Usage examples:
    python train.py --model mlp --task building --epochs 30 --lr 0.001 --batch_size 256
    python train.py --model cnn --task floor    --epochs 50 --lr 0.0005 --batch_size 128
"""

import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data import load_and_preprocess
from models import build_model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MLP or CNN-1D on UJIIndoorLoc."
    )
    parser.add_argument("--model",      type=str,   default="mlp",
                        choices=["mlp", "cnn"],
                        help="Model architecture (default: mlp)")
    parser.add_argument("--task",       type=str,   default="building",
                        choices=["building", "floor"],
                        help="Classification task (default: building)")
    parser.add_argument("--epochs",     type=int,   default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--lr",         type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--batch_size", type=int,   default=256,
                        help="Mini-batch size (default: 256)")
    parser.add_argument("--data_dir",   type=str,   default="data/",
                        help="Directory for dataset CSVs (default: data/)")
    parser.add_argument("--save_dir",   type=str,   default="checkpoints/",
                        help="Directory to save best checkpoint (default: checkpoints/)")
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cuda or cpu (default: auto-detect)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run one epoch. Returns (avg_loss, accuracy)."""
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_batch)

            correct += (logits.argmax(1) == y_batch).sum().item()
            total   += len(y_batch)

    avg_loss = total_loss / total if train else None
    acc = correct / total
    return avg_loss, acc


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seeds(42)

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    X_train, X_val, y_b_train, y_b_val, y_f_train, y_f_val = \
        load_and_preprocess(args.data_dir)

    if args.task == "building":
        y_train, y_val = y_b_train, y_b_val
        num_classes = 3
    else:
        y_train, y_val = y_f_train, y_f_val
        num_classes = 5

    # DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # Clear any leftover CUDA allocations from previous runs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Model, loss, optimiser, scheduler
    model     = build_model(args.model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )

    checkpoint_path = os.path.join(
        args.save_dir, f"{args.model}_{args.task}_best.pt"
    )

    print(
        f"\n[train] model={args.model.upper()}  task={args.task}  "
        f"epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}  "
        f"device={device}  classes={num_classes}"
    )
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Acc':>9}  {'LR':>10}")
    print("-" * 44)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion,
                                  optimizer, device, train=True)
        _, val_acc    = run_epoch(model, val_loader, criterion,
                                  optimizer, device, train=False)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"{epoch:>6}  {train_loss:>11.4f}  {val_acc*100:>8.2f}%  "
            f"{current_lr:>10.6f}"
        )

        scheduler.step()

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_acc": val_acc, "args": vars(args)},
                checkpoint_path,
            )

    print(f"\n[train] Best val accuracy: {best_val_acc*100:.2f}%")
    print(f"[train] Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
