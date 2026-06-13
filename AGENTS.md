# AGENTS.md

## Remote

- `origin`: https://github.com/ivansosabz/flower-classification-transfer-learning.git
- Empty remote; `git fetch origin` before pushing.

## Project structure

```
flower-classification-transfer-learning/
├── src/
│   ├── dataset.py      # DataLoader, augmentations, train/val/test split
│   ├── model.py        # MobileNetV2 backbone + classification head
│   ├── train.py        # Training loop with early stopping by val_acc
│   ├── evaluate.py     # Confusion matrix, F1-score, loss plots
│   └── inference.py    # Single-image prediction (script entrypoint)
├── data/               # ignored (gitignored)
├── models/             # *.pth weights + metric plots (gitignored)
├── notebooks/          # optional exploration
├── requirements.txt
└── .gitignore
```

## Commands

```powershell
# Split raw Kaggle dataset into train/val/test
python src/dataset.py

# Train (freezes backbone, trains classifier head only)
python src/train.py --data-dir data/split --epochs 20 --batch-size 32

# Evaluate on test set
python src/evaluate.py --model-path models/best_model.pth

# Inference on a single image
python src/inference.py path/to/image.jpg --model models/best_model.pth
```

## Key facts

- **Framework**: PyTorch (torchvision). Model weights are `.pth`.
- **Backbone**: MobileNetV2 (pretrained on ImageNet). Features frozen, only classifier head trained.
- **Classes** (5): daisy, dandelion, rose, sunflower, tulip.
- **Input**: 224x224 RGB, normalized with ImageNet mean/std.
- **Augmentation**: RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter.
- **Optimizer**: Adam on `model.classifier.parameters()` only, lr=1e-3.
- **Early stopping**: best model (by val_acc) saved to `models/best_model.pth`.
- **Device**: auto-detects CUDA; falls back to CPU.

## Dataset

- Source: [Kaggle Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) (alxmamaev).
- ~4317 images, ~800 per class.
- Run `prepare_dataset()` in `dataset.py` to split raw folders into `data/split/{train,val,test}` (70/15/15).
