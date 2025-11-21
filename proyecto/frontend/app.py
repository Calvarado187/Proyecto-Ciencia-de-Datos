"""
frontend/app.py
----------------------------------------------------------------------
Aplicación web desarrollada en Streamlit para la demostración del
sistema de reconocimiento de glosas en Lengua de Señas Colombiana (LSC).

Este módulo cumple las siguientes funciones:

1. Cargar el modelo previamente entrenado (lsc_model.h5).
2. Cargar el diccionario de etiquetas (labels.json).
3. Recibir un video corto subido por el usuario.
4. Procesarlo usando el preprocesamiento definido en backend/video_processing.py.
5. Ejecutar la predicción con el modelo.
6. Mostrar la glosa predicha y su nivel de confianza en la interfaz web.

Este archivo constituye el FRONTEND del proyecto, mientras que el
preprocesamiento y modelo se ejecutan en el backend.
----------------------------------------------------------------------
"""

import os
import sys
import json
import tempfile

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS DEL PROYECTO
# ---------------------------------------------------------------------
"""
Creamos la ruta base del proyecto para poder acceder a:
- /backend
- /modelos
Sin depender de la ubicación desde donde se ejecute Streamlit.
"""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Permite importar módulos del backend correctamente
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Importamos la función de preprocesamiento del video
from backend.video_processing import extract_sequence_from_video


# ---------------------------------------------------------------------
# CARGA DEL MODELO Y DICCIONARIO DE ETIQUETAS
# ---------------------------------------------------------------------
"""
Se utiliza @st.cache_resource para que el modelo se cargue una sola vez.
Esto evita tiempos de carga innecesarios cada vez que el usuario
presiona "Predecir glosa".
"""

@st.cache_resource
def load_model_and_labels():
    """
    Carga el modelo entrenado y el diccionario de etiquetas.

    Retorna
    -------
    model : keras.Model
        Modelo RNN previamente entrenado en Google Colab.
    idx2label : dict
        Diccionario con el mapeo índice → glosa.
    """

    # Rutas de modelo y diccionario
    model_path = os.path.join(BASE_DIR, "modelos", "lsc_model.h5")
    labels_path = os.path.join(BASE_DIR, "modelos", "labels.json")

    # Cargar modelo
    model = load_model(model_path)

    # Cargar diccionario glosa → índice y convertirlo a índice → glosa
    with open(labels_path, "r", encoding="utf-8") as f:
        label2idx = json.load(f)

    idx2label = {int(v): k for k, v in label2idx.items()}
    return model, idx2label


# Se cargan una sola vez al inicio
model, idx2label = load_model_and_labels()

# El modelo requiere 16 frames por secuencia
N_FRAMES = 16

# Umbral mínimo de confianza para aceptar la predicción
CONF_THRESH = 0.40


# ---------------------------------------------------------------------
# INTERFAZ DE STREAMLIT
# ---------------------------------------------------------------------
"""
Se define la interfaz gráfica del sistema:
- Título
- Cargador de video
- Vista previa del video
- Botón para predecir
- Salida con glosa + probabilidad
"""

st.set_page_config(page_title="Reconocimiento LSC", layout="centered")

st.title("Reconocimiento de Lengua de Señas Colombiana (LSC)")
st.write(
    "Sube un video corto (~2 segundos) y el modelo intentará reconocer "
    "la **glosa** correspondiente."
)

st.markdown("---")

# Componente de subida de archivos
uploaded_file = st.file_uploader(
    "Sube un video",
    type=["mp4", "mov", "avi", "mkv", "mpeg", "mpg", "mpeg4"],
    help="Formatos soportados: MP4, MOV, AVI, MKV, MPEG, etc.",
)

temp_path = None

# -------------------------------------------------------------
# MANEJO DEL VIDEO SUBIDO
# -------------------------------------------------------------
if uploaded_file is not None:
    # Mostrar video en la interfaz
    st.video(uploaded_file)

    # Guardar el archivo en un temporal (OpenCV solo lee rutas)
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Botón de predicción
    if st.button("Predecir glosa"):

        if temp_path is None:
            st.error("No se pudo guardar el video temporalmente.")
        else:
            with st.spinner("Procesando video y obteniendo predicción..."):

                try:
                    # -----------------------------------------------------
                    # 1. EXTRAER SECUENCIA DE CARACTERÍSTICAS DEL VIDEO
                    # -----------------------------------------------------
                    seq = extract_sequence_from_video(
                        temp_path, num_frames=N_FRAMES
                    )

                    # seq es (16, 150) → añadimos dimensión batch → (1, 16, 150)
                    seq = np.expand_dims(seq, axis=0)

                    # -----------------------------------------------------
                    # 2. PREDICCIÓN DEL MODELO
                    # -----------------------------------------------------
                    preds = model.predict(seq)[0]  # vector (num_clases,)

                    best_idx = int(np.argmax(preds))
                    best_prob = float(preds[best_idx])
                    best_gloss = idx2label.get(best_idx, f"clase_{best_idx}")

                    # -----------------------------------------------------
                    # 3. VALIDACIÓN POR UMBRAL DE CONFIANZA
                    # -----------------------------------------------------
                    if best_prob < CONF_THRESH:
                        st.warning(
                            f"No estoy seguro de la glosa. "
                            f"Mejor candidato: **{best_gloss}** "
                            f"(confianza: {best_prob:.2f}).\n\n"
                            "El modelo necesita más entrenamiento o un video más claro."
                        )
                    else:
                        st.success(
                            f"Glosa predicha: **{best_gloss}** "
                            f"(confianza: {best_prob:.2f})"
                        )

                except Exception as e:
                    # Captura de errores en predicción
                    st.error(f"Ocurrió un error durante la predicción: {e}")

            # Intentar borrar el video temporal
            try:
                os.remove(temp_path)
            except Exception:
                pass
