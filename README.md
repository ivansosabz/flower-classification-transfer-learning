# Clasificacion de Flores

Este proyecto usa inteligencia artificial para reconocer 5 tipos de flores en fotos: **rosa, tulipan, margarita, diente de leon y girasol**.

## Requisitos

- **Python 3.12** instalado ([descargar](https://www.python.org/downloads/))
- Conexion a internet (para instalar dependencias y descargar las fotos)

## Instalacion paso a paso

### 1. Descargar el proyecto

**Opcion A (con Git, recomendado):**

```powershell
git clone https://github.com/ivansosabz/flower-classification-transfer-learning.git
cd flower-classification-transfer-learning
```

**Opcion B (sin Git):** Descargar el ZIP desde GitHub, extraerlo y abrir la carpeta en PowerShell.

### 2. Crear el entorno virtual

Un entorno virtual es una cajita aislada donde se instalan las librerias del proyecto, sin mezclarlas con otros proyectos.

```powershell
py -3.12 -m venv .venv
```

Para activarlo:

```powershell
.\.venv\Scripts\Activate.ps1
```

> Si da error de ejecucion en PowerShell, ejecutar antes: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

Una vez activado, vas a ver `(.venv)` al principio de la linea en la terminal.

### 3. Instalar las dependencias

```powershell
pip install -r requirements.txt
```

Esto instala PyTorch (el motor de inteligencia artificial), torchvision (para procesar imagenes), y otras librerias necesarias.

### 4. Descargar el dataset (las fotos de flores)

El dataset original esta en Kaggle: [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

```powershell
# Descargar directo desde Kaggle (si tenes kaggle CLI):
pip install kaggle
kaggle datasets download alxmamaev/flowers-recognition -p data/

# Si no tenes kaggle, descargar manualmente:
# 1. Ir a https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
# 2. Hacer click en "Download"
# 3. Colocar flowers-recognition.zip en la carpeta data/
```

Extraer el archivo:

```powershell
Expand-Archive -Path "data/flowers-recognition.zip" -DestinationPath "data/raw"
```

### 5. Dividir las fotos en entrenamiento, validacion y prueba

```powershell
python -c "from src.dataset import prepare_dataset; prepare_dataset('data/raw', 'data/split')"
```

Esto separa las fotos en:
- **train** (70%): para que el modelo aprenda
- **val** (15%): para ajustar el modelo durante el entrenamiento
- **test** (15%): para medir el resultado final

### 6. Entrenar el modelo

```powershell
python src/train.py --data-dir data/split --epochs 20 --batch-size 32
```

Esto entrena la inteligencia artificial. Va a tardar entre 15 y 30 minutos en una laptop normal (CPU). Cuando termina, guarda los pesos del modelo en `models/best_model.pth`.

> Si ya tenes el modelo entrenado (el archivo `best_model.pth`), podes saltear este paso.

### 7. Evaluar el modelo

```powershell
python src/evaluate.py --model-path models/best_model.pth
```

Esto prueba el modelo con las fotos de test (las que nunca vio) y muestra:
- **Accuracy** (porcentaje de aciertos): nuestro modelo actual tiene **88%**
- **F1-Score** (combinacion de precision y efectividad)
- **Matriz de confusion** (una tabla que muestra con que flores se confunde)

### 8. Probar con tu propia foto

```powershell
python src/inference.py ruta/completa/de/tu/foto.jpg --model models/best_model.pth
```

Reemplaza `ruta/completa/de/tu/foto.jpg` por la ruta de una foto de flor que tengas. El programa va a decir:

```
Prediccion: rose
Confianza: 0.9234 (92.3%)
```

Para probar con una foto del dataset:

```powershell
python src/inference.py "data/split/test/rose/16793843266_43c56fa890_n.jpg" --model models/best_model.pth
```

### 9. Abrir el notebook (opcional, para presentar)

```powershell
pip install jupyter
jupyter notebook notebooks/flower_classification.ipynb
```

Esto abre un archivo interactivo con todo el proyecto explicado visualmente: fotos de ejemplo, graficos, resultados. Ideal para mostrarlo en la defensa.

## Estructura del proyecto

| Carpeta | Contiene |
|---|---|
| `src/` | El codigo fuente (5 archivos .py) |
| `data/` | Las fotos del dataset (no se sube a GitHub) |
| `models/` | Los pesos del modelo entrenado y graficos (no se sube a GitHub) |
| `notebooks/` | Notebook de Jupyter para la presentacion |

## Dataset

**Flowers Recognition** (alxmamaev) - [Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- 5 clases: daisy, dandelion, rose, sunflower, tulip
- ~4317 imagenes a color
- ~800 fotos por clase

## Creditos

- Daniela Beatriz Gonzalez Aguilera - Ing. en Electronica
- Jesus Ivan Sosa Baez - Ing. en Sistemas Informaticos
- Vision Artificial Aplicada - Segundo Examen Parcial (Fase 1)
