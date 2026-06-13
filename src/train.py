# train.py
#
# PROPOSITO: Entrena el modelo para que aprenda a clasificar flores.
# TECNOLOGIA: PyTorch, Adam, tqdm.
#
# QUE HACE:
# 1. Carga las fotos de train y val usando get_dataloaders()
# 2. Crea el modelo MobileNetV2 con build_model()
# 3. Por cada EPOCA (vuelta completa a todas las fotos):
#    a. train_epoch() -> pasa las fotos de entrenamiento por el modelo,
#       calcula el error (loss), ajusta los pesos con Adam
#    b. validate() -> pasa las fotos de validacion SIN entrenar, solo
#       mide que tan bien le esta yendo
# 4. Si el accuracy de validacion mejora, guarda los pesos en best_model.pth
#    (early stopping: nos quedamos con el mejor modelo)
#
# SOLO se entrenan las capas del classifier, el backbone queda congelado.
#
# EJECUCION:
#   python src/train.py --data-dir data/split --epochs 20 --batch-size 32

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model


def train_epoch(model, loader, criterion, optimizer, device):
    # Modo entrenamiento: activa dropout y permite ajustar pesos
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()           # limpia ajustes anteriores
        outputs = model(images)          # pasa las fotos por el modelo
        loss = criterion(outputs, labels)  # calcula el error
        loss.backward()                  # calcula como ajustar los pesos
        optimizer.step()                 # aplica el ajuste

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    # Modo evaluacion: desactiva dropout, no ajusta pesos
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # no calcula gradientes (ahorra memoria y tiempo)
        for images, labels in tqdm(loader, desc="Val"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )

    model = build_model(num_classes=5, dropout_rate=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()   # funcion de error para clasificacion
    optimizer = Adam(model.classifier.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoca {epoch:2d}/{args.epochs}  "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Si mejoro, guardamos el modelo (early stopping)
        if val_acc > best_acc:
            best_acc = val_acc
            models_dir = Path(args.models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), models_dir / "best_model.pth")
            print(f"  -> Guardado mejor modelo (val_acc={val_acc:.4f})")

    print(f"Entrenamiento completo. Mejor val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/split")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()
    main(args)
