"""
train.py — Full Training Loop for Food Quality Classifier

Run:
    python train.py --data_root ./data/food --epochs 15 --batch_size 32

Key design decisions (memorize for oral defense!):
- Loss: CrossEntropyLoss with class weights → handles fresh/rotten imbalance
- Optimizer: AdamW → weight decay decoupled from gradient update
- Scheduler: CosineAnnealingLR → smooth LR decay, avoids overshooting
- Early stopping → saves best model, prevents overfitting
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, balanced_accuracy_score

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.dataset import build_dataloaders
from models.model import build_model, count_trainable_params


# ── Argument parser ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train Food Quality Classifier")
    parser.add_argument("--data_root",  type=str,   default="./data/food")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--lr_head",    type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=5)
    parser.add_argument("--save_path",  type=str,   default="./best_model.pth")
    return parser.parse_args()


# ── Class weights ──────────────────────────────────────────────────────────────
def compute_class_weights(train_loader, num_classes: int, device):
    """
    Computes inverse-frequency weights.
    If dataset has 1000 fresh and 500 rotten images:
    → rotten gets 2x higher weight in the loss
    → model can't just predict 'fresh' every time and get good loss
    """
    counts = torch.zeros(num_classes)
    for _, labels in train_loader:
        for l in labels:
            counts[l] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    print(f"Class weights: Fresh={weights[0]:.3f}, Rotten={weights[1]:.3f}")
    return weights.to(device)


# ── Train one epoch ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # Clear old gradients
        logits = model(images)         # Forward pass
        loss   = criterion(logits, labels)  # Compute loss
        loss.backward()                # Backpropagation
        optimizer.step()               # Update weights

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ── Validation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / total, correct / total, bal_acc, all_preds, all_labels


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🍎 Food Quality Classifier — Training")
    print(f"Device : {device}")
    print(f"Data   : {args.data_root}")
    print(f"Epochs : {args.epochs}\n")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_loader, val_loader, class_names = build_dataloaders(
        args.data_root, batch_size=args.batch_size
    )
    num_classes = len(class_names)

    # ── Model ──────────────────────────────────────────────────────────────────
    model    = build_model(num_classes=num_classes).to(device)
    n_params = count_trainable_params(model)
    print(f"Trainable parameters: {n_params:,}\n")

    # ── Loss ───────────────────────────────────────────────────────────────────
    class_weights = compute_class_weights(train_loader, num_classes, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer ──────────────────────────────────────────────────────────────
    # Different learning rates: lower for backbone, higher for new head
    optimizer = AdamW([
        {"params": model.features[-3:].parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(),    "lr": args.lr_head},
    ], weight_decay=1e-4)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, bal_acc, preds, labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
            f"Balanced Acc: {bal_acc:.3f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "class_names": class_names,
            }, args.save_path)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")

        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹ Early stopping at epoch {epoch}")
                break

    # ── Final report ───────────────────────────────────────────────────────────
    print("\n── Final Validation Report ──")
    print(classification_report(
        labels, preds,
        target_names=class_names,
        zero_division=0
    ))
    print(f"Best Val Loss : {best_val_loss:.4f}")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()