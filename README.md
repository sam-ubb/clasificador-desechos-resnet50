# Clasificador de Desechos con Transfer Learning (ResNet50)

Proyecto desarrollado para el **Taller de Introducción a Visión por Computadora**

Clasificador de imágenes de residuos en tres categorías: **cartón (cardboard)**,
**metal** y **plástico (plastic)**, construido mediante *Transfer Learning* sobre la
arquitectura **ResNet50** preentrenada en ImageNet.

## Descripción del proyecto

El objetivo es clasificar automáticamente una imagen de un residuo en una de las tres
clases mencionadas. El proyecto se desarrolló de forma incremental a lo largo de varios
avances:

- **Preparación del dataset:** las imágenes provienen de un dataset gestionado en Roboflow,
  organizado en particiones `train`, `valid` y `test`. Se aplicó *data augmentation*
  únicamente al conjunto de entrenamiento (volteo horizontal, rotación y ajustes de
  brillo/exposición).
- **Modelo:** ResNet50 con pesos de ImageNet. Se congelaron las capas iniciales y se
  reentrenaron las capas finales (`layer4` y la capa `fc`) adaptando la salida a 3 clases.
- **Entrenamiento y selección de modelo:** el checkpoint se guarda según el desempeño en el
  conjunto de **validación**, y el conjunto de **test** se utiliza una sola vez al final
  para la evaluación, evitando fuga de información.
- **Evaluación:** matriz de confusión, reporte de clasificación por clase, curvas
  Precision-Recall y análisis de falsos positivos/negativos.
- **Pipeline de inferencia (Avance 4):** carga del modelo entrenado e inferencia sobre
  imágenes nuevas, dibujando la etiqueta y la confianza sobre la imagen **usando
  exclusivamente OpenCV**.

## Estructura del repositorio

```
.
├── ClasificadorDeDesechos.ipynb   # Notebook principal (código completo)
├── README.md                      # Este archivo
├── imagenes_nuevas/               # Imágenes de prueba (fuera del dataset original)
└── resultados/                    # Salidas de inferencia con la etiqueta dibujada
```

## Librerías necesarias

El proyecto está pensado para ejecutarse en **Google Colab** (con GPU recomendada).
Las principales dependencias son:

- `torch` y `torchvision` — modelo ResNet50 y entrenamiento
- `opencv-python` (`cv2`) — dibujo de resultados en el pipeline de inferencia
- `numpy` — manejo de arreglos
- `Pillow` (`PIL`) — carga de imágenes
- `matplotlib` — visualización dentro del notebook
- `scikit-learn` — matriz de confusión, reporte de clasificación y curvas PR
- `seaborn` — visualización de la matriz de confusión
- `roboflow` — descarga del dataset

Instalación (si se ejecuta fuera de Colab):

```bash
pip install torch torchvision opencv-python numpy pillow matplotlib scikit-learn seaborn roboflow
```

## Cómo ejecutar

1. Abrir `ClasificadorDeDesechos.ipynb` en Google Colab.
2. Ejecutar las celdas en orden (**Entorno de ejecución → Ejecutar todo**). Esto descarga
   el dataset, entrena el modelo y genera las métricas de evaluación.
3. Para la inferencia sobre imágenes nuevas, colocar las imágenes en la carpeta
   `imagenes_nuevas/` y ejecutar las celdas del pipeline de inferencia. Los resultados se
   guardan en `resultados/`.

## Evaluación con datos nuevos

El modelo fue probado con imágenes que no pertenecen al dataset original, incluyendo
objetos individuales sobre fondo simple y escenas complejas del mundo real (playas,
vertederos, chatarra). Se observó un buen desempeño en objetos individuales y una
degradación notable en escenas complejas fuera de la distribución de entrenamiento
(*domain shift*), junto con sobreconfianza en las predicciones erróneas. El análisis
crítico completo se encuentra en el notebook.

## Autores

- Samuel Armando Fuentes Durán
- Cristian Alejandro Torres Medina

Taller de Introducción a Visión por Computadora — 2026.
