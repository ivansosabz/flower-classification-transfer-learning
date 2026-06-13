# model.py
#
# PROPOSITO: Define la red neuronal que va a clasificar las flores.
# TECNOLOGIA: PyTorch, torchvision (MobileNetV2).
#
# QUE HACE:
# build_model() -> crea el modelo completo con dos partes:
#   1. BACKBONE (MobileNetV2): un modelo que ya fue entrenado con millones de
#      fotos (ImageNet). Sabe reconocer bordes, formas, colores. Lo dejamos
#      CONGELADO (no se entrena) porque ya sabe suficiente.
#   2. CLASSIFIER (cabeza de clasificacion): capas nuevas que reemplazan a la
#      cabeza original. Estas SI se entrenan para aprender a distinguir las
#      5 flores nuestras.
#
# ARQUITECTURA DEL CLASSIFIER:
#   Dropout(0.3)  -> apaga el 30% de las neuronas al azar (evita sobreajuste)
#   Linear(1280,512)  -> capa que conecta 1280 numeros a 512
#   ReLU  -> funcion que deja pasar solo numeros positivos
#   Dropout(0.3)  -> otra capa de apagado
#   Linear(512,5)  -> capa final que da 5 numeros (uno por cada flor)
#
# El numero mas alto de los 5 finales define que flor es.

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=5, dropout_rate=0.3):
    # Cargamos MobileNetV2 ya entrenado en ImageNet
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Congelamos el backbone (no se va a entrenar)
    for param in model.features.parameters():
        param.requires_grad = False

    # Reemplazamos la cabeza por una nueva para nuestras 5 clases
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes),
    )

    return model


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
