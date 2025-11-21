# Proyecto-Ciencia-de-Datos
Proyecto de ciencia de datos 
Proyecto: Reconocimiento de Lengua de Señas Colombiana (LSC)

Este proyecto implementa un prototipo funcional para el reconocimiento de glosas de la Lengua de Señas Colombiana (LSC) a partir de videos cortos.
El sistema procesa un video, extrae características cuadro a cuadro y utiliza un modelo de Deep Learning para predecir la glosa correspondiente.

1. Estructura del Proyecto
Proyecto-Ciencia-de-Datos/
│
├── backend/
│   └── video_processing.py
│
├── frontend/
│   └── app.py
│
├── modelos/
│   ├── lsc_model.h5
│   └── labels.json
│
├── requirements.txt
└── README.md

2. Descripción General

El sistema permite:

Cargar un video desde la interfaz web.

Extraer una secuencia de características numéricas.

Procesar la secuencia mediante un modelo entrenado previamente.

Mostrar la glosa predicha y el nivel de confianza.

El objetivo principal es demostrar el funcionamiento completo del pipeline, desde el procesamiento del video hasta la predicción final.

3. Tecnologías Utilizadas

Python 3.10

TensorFlow 2.x

Streamlit

OpenCV

NumPy

4. Ejecución del Proyecto
1. Crear entorno virtual
python -m venv venv

2. Activarlo

Windows:

venv\Scripts\activate

3. Instalar dependencias
pip install -r requirements.txt

4. Ejecutar la aplicación
streamlit run frontend/app.py

5. Limitaciones

Modelo entrenado con pocas muestras.

Extracción de características simple (píxeles en escala de grises).

Las predicciones pueden ser imprecisas y requieren mayor entrenamiento.

6. Trabajo Futuro

Ampliar el dataset.

Implementar extracción avanzada con MediaPipe.

Entrenar modelos más complejos (CNN+LSTM, Transformers).

Mejorar la interfaz y publicar el sistema en la nube.
