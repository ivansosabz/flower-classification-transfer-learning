# Carpeta `models/`

## Que contiene

Los archivos que se generan al entrenar el modelo.

## Archivos

| Archivo | Que es |
|---|---|
| `best_model.pth` | Los pesos del modelo ya entrenado (11.7 MB). Se genera con `train.py`. |
| `confusion_matrix.png` | Grafico de la matriz de confusion (resultados en test). Se genera con `evaluate.py`. |

## Notas

- Esta carpeta esta en `.gitignore` -- no se sube a GitHub.
- `.pth` es el formato de PyTorch para guardar los pesos de la red.
- Para usar el modelo: `inference.py` carga estos pesos y predice flores nuevas.
