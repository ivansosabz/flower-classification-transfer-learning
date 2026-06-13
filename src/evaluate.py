import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay

from dataset import get_dataloaders, CLASSES
from model import build_model


def plot_loss(history, save_path="models/loss_plot.png"):
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)

    model = build_model(num_classes=5).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    accuracy = (all_preds == all_labels).mean()

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print("\nPer-class F1-Score:")
    for cls_name, f1 in zip(CLASSES, f1_per_class):
        print(f"  {cls_name:12s}: {f1:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (Acc={accuracy:.4f}, F1={f1_macro:.4f})")
    plt.savefig(args.cm_path)
    plt.close()
    print(f"Confusion matrix saved to {args.cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/split")
    parser.add_argument("--model-path", type=str, default="models/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cm-path", type=str, default="models/confusion_matrix.png")
    args = parser.parse_args()
    evaluate(args)
