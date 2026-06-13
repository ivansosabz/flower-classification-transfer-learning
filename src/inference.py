# inference.py
#
# PROPOSITO: Toma una foto de una flor y predice de que tipo es.
# TECNOLOGIA: PyTorch, PIL.
#
# QUE HACE:
# predict() -> recibe la ruta de una imagen, la procesa (redimensiona a
# 224x224, normaliza colores), la pasa por el modelo entrenado, y devuelve
# el nombre de la flor + el nivel de confianza (0 a 1).
#
# Se usa desde consola:
#   python src/inference.py ruta/de/una/foto.jpg --model models/best_model.pth
#
# La funcion predict() tambien se puede importar desde otro script (util
# para la Fase 2 cuando hagamos la interfaz grafica).

import argparse
from pathlib import Path

import torch
from PIL import Image

from dataset import CLASSES, IMG_SIZE, MEAN, STD, val_transform
from model import build_model


def predict(image_path, model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    input_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Softmax convierte los numeros crudos en probabilidades (suma = 1)
    class_name = CLASSES[predicted.item()]
    confidence = confidence.item()

    return class_name, confidence


def main():
    parser = argparse.ArgumentParser(description="Inferencia sobre una imagen de flor")
    parser.add_argument("image_path", type=str, help="Ruta a la imagen")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Ruta a los pesos del modelo")
    args = parser.parse_args()

    if not Path(args.image_path).is_file():
        print(f"Error: no se encuentra la imagen '{args.image_path}'")
        return
    if not Path(args.model).is_file():
        print(f"Error: no se encuentra el modelo '{args.model}'")
        return

    class_name, confidence = predict(args.image_path, args.model)
    print(f"Prediccion: {class_name}")
    print(f"Confianza: {confidence:.4f} ({confidence * 100:.1f}%)")


if __name__ == "__main__":
    main()
