# evaluate.py
#
# PROPOSITO: Mide que tan bien funciona el modelo entrenado.
# TECNOLOGIA: PyTorch, scikit-learn, matplotlib.
#
# QUE HACE:
# 1. Carga las fotos de TEST (las que el modelo NUNCA vio durante entrenamiento)
# 2. Carga el modelo guardado (best_model.pth)
# 3. Pasa todas las fotos de test por el modelo y compara con la respuesta correcta
# 4. Calcula y muestra:
#    - Accuracy (porcentaje de aciertos)
#    - F1-Score macro (promedio de precision+recall por clase)
#    - F1-Score por cada flor
#    - Matriz de confusion (grafico que muestra donde se confunde el modelo)
# 5. Guarda la matriz de confusion como imagen PNG
#
# EJECUCION:
#   python src/evaluate.py --model-path models/best_model.pth

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
    plt.xlabel("Epoca")
    plt.ylabel("Perdida (Loss)")
    plt.legend()
    plt.title("Perdida durante entrenamiento")
    plt.savefig(save_path)
    plt.close()
    print(f"Grafico de perdida guardado en {save_path}")


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

    # Calculo de metricas
    cm = confusion_matrix(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    accuracy = (all_preds == all_labels).mean()

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print("\nF1-Score por clase:")
    for cls_name, f1 in zip(CLASSES, f1_per_class):
        print(f"  {cls_name:12s}: {f1:.4f}")

    # Grafico de matriz de confusion
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Matriz de Confusion (Acc={accuracy:.4f}, F1={f1_macro:.4f})")
    plt.savefig(args.cm_path)
    plt.close()
    print(f"Matriz de confusion guardada en {args.cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/split")
    parser.add_argument("--model-path", type=str, default="models/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cm-path", type=str, default="models/confusion_matrix.png")
    args = parser.parse_args()
    evaluate(args)
