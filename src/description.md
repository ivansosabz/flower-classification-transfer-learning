# Carpeta `src/`

## Que contiene

El codigo fuente del proyecto, separado en 5 archivos segun su responsabilidad.

## Archivos

### 1. `dataset.py`
Prepara las fotos y las carga para entrenar. Divide el dataset en train/val/test.
Usa PyTorch + torchvision.

### 2. `model.py`
Define la red neuronal: MobileNetV2 pre-entrenado + cabeza de clasificacion nueva.
Usa PyTorch + torchvision.

### 3. `train.py`
Entrena el modelo. Pasa las fotos de train, calcula el error, ajusta los pesos,
y guarda el mejor modelo. Usa PyTorch + Adam.

### 4. `evaluate.py`
Evalua el modelo entrenado con las fotos de test. Muestra accuracy, F1-score,
y genera la matriz de confusion. Usa scikit-learn + matplotlib.

### 5. `inference.py`
Predice una foto nueva. Toma una imagen, la procesa, la pasa por el modelo,
y dice que flor es con que tan seguro esta. Usa PyTorch + PIL.

## Orden de ejecucion

```
1. dataset.py (preparar datos)  -> 2. train.py (entrenar)
                                  -> 3. evaluate.py (medir resultado)
                                  -> 4. inference.py (predecir una foto)
```

## Tecnologias

- Python 3.12
- PyTorch (torch, torchvision)
- scikit-learn, matplotlib, PIL, tqdm
