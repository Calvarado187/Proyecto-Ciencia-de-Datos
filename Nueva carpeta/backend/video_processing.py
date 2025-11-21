# backend/video_processing.py

import cv2
import numpy as np

# Número de características por frame que espera el modelo
# (tu modelo entrenado tiene entrada (16, 150))
N_FEATURES = 150


def _extract_features_from_frame(image) -> np.ndarray:
    """
    Convierte un frame a escala de grises, lo redimensiona y lo aplana
    en un vector de longitud N_FEATURES.
    """
    # Escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Redimensionar a 10 x 15 = 150 píxeles
    resized = cv2.resize(gray, (15, 10))  # (ancho, alto)

    # Normalizar a [0,1]
    norm = resized.astype(np.float32) / 255.0

    # Aplanar a vector 1D de longitud 150
    flat = norm.flatten()  # (150,)

    return flat


def extract_sequence_from_video(video_path: str, num_frames: int = 16) -> np.ndarray:
    """
    Lee un video desde 'video_path' y devuelve una secuencia de tamaño
    (num_frames, N_FEATURES), lista para pasar al modelo.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        # Si no se pudo abrir el video, devolvemos todo ceros
        return np.zeros((num_frames, N_FEATURES), dtype=np.float32)

    frames_features = []

    # Leer todos los frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        feats = _extract_features_from_frame(frame)
        frames_features.append(feats)

    cap.release()

    # Si no se extrajo nada, devolvemos todo ceros
    if len(frames_features) == 0:
        return np.zeros((num_frames, N_FEATURES), dtype=np.float32)

    frames_features = np.array(frames_features, dtype=np.float32)

    # Ajustar número de frames a num_frames
    if len(frames_features) >= num_frames:
        # Elegimos num_frames frames distribuidos a lo largo del video
        indices = np.linspace(0, len(frames_features) - 1, num_frames).astype(int)
        seq = frames_features[indices]
    else:
        # Si hay menos frames, repetimos el último hasta completar
        last = frames_features[-1]
        pad = np.repeat(last[None, :], num_frames - len(frames_features), axis=0)
        seq = np.concatenate([frames_features, pad], axis=0)

    return seq
