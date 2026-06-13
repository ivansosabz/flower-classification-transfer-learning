# Flower Classification with Transfer Learning

Clasificacion multiclase de imagenes de flores en 5 categorias (rosa, tulipan, margarita, diente de leon, girasol) usando MobileNetV2 con Transfer Learning en PyTorch.

## Dataset

[Kaggle Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) - Alexander Mamaev.
- 5 clases: daisy, dandelion, rose, sunflower, tulip
- ~4317 imagenes, ~800 por clase

## Requisitos

```
pip install -r requirements.txt
```

## Uso

### 1. Preparar dataset

Descargar el dataset de Kaggle y extraerlo en `data/raw/`. Luego:

```python
from src.dataset import prepare_dataset
prepare_dataset("data/raw", "data/split")
```

### 2. Entrenar

```bash
python src/train.py --data-dir data/split --epochs 20
```

### 3. Evaluar

```bash
python src/evaluate.py --model-path models/best_model.pth
```

### 4. Inferencia

```bash
python src/inference.py ruta/a/imagen.jpg --model models/best_model.pth
```

## Arquitectura

- **Backbone**: MobileNetV2 pre-entrenado en ImageNet (capas congeladas)
- **Classifier**: Dropout(0.3) -> Linear(1280, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 5)
- **Optimizer**: Adam (lr=1e-3)
- **Data Augmentation**: RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter

## Metricas

- Accuracy, Macro F1-Score, F1 por clase
- Matriz de confusion
- Grafico de perdida (loss) por epoca

## Entregables

- Codigo fuente modular en `src/`
- Pesos del modelo: `models/best_model.pth`
- Link al dataset original (ver seccion Dataset)

## Creditos

- Daniela Beatriz Gonzalez Aguilera - Ing. en Electronica
- Jesus Ivan Sosa Baez - Ing. en Sistemas Informaticos
- Vision Artificial Aplicada - Segundo Examen Parcial
