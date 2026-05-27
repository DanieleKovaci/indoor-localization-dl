"""
evaluate.py — Load trained checkpoints and print a comparison table.

The bachelor thesis (Kovaci, 2026) used SPSS linear regression to analyse factors
affecting positioning error (Euclidean distance in metres) on a 100-sample subset of
UJIIndoorLoc. It did not perform building/floor classification, so no direct accuracy
baseline exists. The thesis key result is reported below for context.
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from data import load_and_preprocess
from models import build_model

# ── Thesis context (Kovaci 2026) ─────────────────────────────────────────────
# Task:   SPSS linear regression predicting positioning error (metres)
# Result: best model R² = 0.094 (explains only 9.4% of error variance)
#         mean positioning error = 115.88 m, range 4.43–263.19 m
# Conclusion: RSSI-only linear models have very limited predictive power;
#             more advanced models are needed.
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained checkpoints and print comparison table."
    )
    parser.add_argument("--data_dir",   type=str, default="data/")
    parser.add_argument("--save_dir",   type=str, default="checkpoints/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict(model, loader, device):
    """Return (all_preds, all_labels) as numpy arrays."""
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds_list.append(logits.argmax(1).cpu().numpy())
            labels_list.append(y_batch.numpy())
    return np.concatenate(preds_list), np.concatenate(labels_list)


def accuracy_pct(preds, labels):
    return 100.0 * np.mean(preds == labels)


def per_class_accuracy(preds, labels):
    """Returns dict {class_id: accuracy_pct}."""
    result = {}
    for cls in np.unique(labels):
        mask = labels == cls
        result[cls] = 100.0 * np.mean(preds[mask] == cls)
    return result


def confusion_matrix_text(preds, labels, num_classes):
    """Plain-text confusion matrix — no matplotlib."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(preds, labels):
        cm[t, p] += 1

    col_width = max(6, len(str(cm.max())) + 2)
    header = " " * 8 + "".join(f"Pred {c:>{col_width - 5}}" for c in range(num_classes))
    lines = [header, "-" * len(header)]
    for i in range(num_classes):
        row = f"True {i:2d}  " + "".join(f"{cm[i, j]:>{col_width}}" for j in range(num_classes))
        lines.append(row)
    return "\n".join(lines)


def load_checkpoint(checkpoint_path, model_name, num_classes, device):
    """Load a saved checkpoint and return the model in eval mode."""
    model = build_model(model_name, num_classes).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device(args.device)

    X_train, X_val, y_b_train, y_b_val, y_f_train, y_f_val = \
        load_and_preprocess(args.data_dir)

    configs = [
        ("building", 3, y_b_val),
        ("floor",    5, y_f_val),
    ]

    results = {}

    for task, num_classes, y_val in configs:
        val_ds     = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        results[task] = {}

        for model_name in ("mlp", "cnn"):
            ckpt_path = os.path.join(args.save_dir, f"{model_name}_{task}_best.pt")

            if not os.path.exists(ckpt_path):
                print(f"[eval] Checkpoint not found: {ckpt_path} — skipping.")
                results[task][model_name] = None
                continue

            model = load_checkpoint(ckpt_path, model_name, num_classes, device)
            preds, labels = predict(model, val_loader, device)
            acc = accuracy_pct(preds, labels)
            results[task][model_name] = acc

            # Per-class accuracy
            pc_acc = per_class_accuracy(preds, labels)
            label_name = "Building" if task == "building" else "Floor"
            print(f"\n--- {model_name.upper()} / {task} ---")
            print(f"Overall accuracy: {acc:.2f}%")
            print(f"Per-{label_name.lower()} accuracy:")
            for cls, cls_acc in pc_acc.items():
                print(f"  {label_name} {cls}: {cls_acc:.2f}%")

            # Confusion matrix
            print(f"Confusion matrix ({task}):")
            print(confusion_matrix_text(preds, labels, num_classes))

    # ── Final comparison table ───────────────────────────────────────────────
    _divider = "=" * 52

    for task, num_classes, _ in configs:
        mlp_str = f"{results[task]['mlp']:.2f}%"  if results[task].get("mlp")  is not None else "N/A"
        cnn_str = f"{results[task]['cnn']:.2f}%"  if results[task].get("cnn")  is not None else "N/A"

        print(f"\n{_divider}")
        print(f"  Results: {task.capitalize()} Classification")
        print(_divider)
        print(f"  {'Method':<22}{'Val Accuracy'}")
        print(f"  {'-'*20}  {'-'*14}")
        print(f"  {'MLP':<22}{mlp_str}")
        print(f"  {'CNN-1D':<22}{cnn_str}")

    print(f"\n  Thesis baseline (Kovaci 2026): SPSS linear regression on 100-sample")
    print(f"  subset, predicting positioning error. Best R² = 0.094 (9.4% variance")
    print(f"  explained). Task is regression, not classification — not directly")
    print(f"  comparable, but confirms that RSSI-only linear models have very")
    print(f"  limited predictive power on this dataset.")

    print()


if __name__ == "__main__":
    main()
