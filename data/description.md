# Carpeta `data/`

## Que contiene

Las fotos del dataset **Flowers Recognition** de Kaggle (alxmamaev).

## Estructura

```
data/
├── raw/                       # fotos originales (una carpeta por cada flor)
│   ├── daisy/                 # ~764 fotos de margaritas
│   ├── dandelion/             # ~1052 fotos de dientes de leon
│   ├── rose/                  # ~784 fotos de rosas
│   ├── sunflower/             # ~733 fotos de girasoles
│   └── tulip/                 # ~984 fotos de tulipanes
├── split/                     # fotos separadas para entrenar y evaluar
│   ├── train/                 # 70% de las fotos (para entrenar)
│   ├── val/                   # 15% de las fotos (para validar durante entrenamiento)
│   └── test/                  # 15% de las fotos (para medir resultado final)
└── flowers-recognition.zip    # archivo original descargado de Kaggle
```

## Notas

- Esta carpeta esta en `.gitignore` -- no se sube a GitHub porque pesa mucho.
- Para generarla: descargar de Kaggle, extraer en `data/raw/`, y ejecutar
  `prepare_dataset("data/raw", "data/split")` desde Python.
- Las fotos son a color (RGB), resolucion variable (~320x240 promedio).
