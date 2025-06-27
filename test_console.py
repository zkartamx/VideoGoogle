import os
import sys

# Añadir la ruta del proyecto al sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.model_loader import load_videoprism_video_model
from src.video_processor import extract_clips, get_video_duration
from src.embedding_extractor import extract_video_embeddings
from src.scene_detector import find_scene_cuts

def run_test():
    """Runs a console-based test of the video processing logic."""
    print("--- INICIANDO PRUEBA DE CONSOLA ---")

    # --- Configuración ---
    # Asegúrate de que esta ruta apunte a un vídeo real en tu sistema
    video_path = os.path.join(project_root, "input_videos", "video3.MP4") 
    threshold = 0.98
    resolution = 288

    if not os.path.exists(video_path):
        print(f"ERROR: El video de prueba no se encuentra en {video_path}")
        print("Por favor, coloca un fichero llamado 'test_video.mp4' en el directorio del proyecto.")
        return

    # --- Carga de Modelo ---
    print("Cargando modelo de video...")
    video_model_fn = load_videoprism_video_model(model_name='base', use_fp16=True)
    print("Modelo cargado.")

    # --- Procesamiento ---
    print(f"1. Extrayendo clips de '{video_path}' con resolución {resolution}p...")
    clips = extract_clips(video_path, resolution=resolution)
    if not clips:
        print("ERROR: No se pudieron extraer clips.")
        return
    print(f"Se extrajeron {len(clips)} clips.")

    print("Calculando la duración del video...")
    duration = get_video_duration(video_path)
    clip_duration = duration / len(clips) if clips else 0
    print(f"Duración del video: {duration:.2f}s, Duración por clip: {clip_duration:.2f}s")

    print("2. Extrayendo embeddings de los clips...")
    video_embeddings = extract_video_embeddings(clips, video_model_fn)
    print(f"Se extrajeron {len(video_embeddings)} embeddings.")

    print(f"3. Detectando cortes de escena con umbral = {threshold}...")
    cut_timestamps, similarities, _ = find_scene_cuts(video_path, video_embeddings, threshold, clip_duration)

    # --- Detalle de Similitudes ---
    print("\n--- DETALLE DE SIMILITUDES ---")
    if similarities:
        for i, sim in enumerate(similarities):
            print(f"Clip {i}→{i+1}: sim = {sim:.3f}")
    else:
        print("No se calcularon similitudes.")
    print("--------------------------------")

    # --- Resultados ---
    print("\n--- RESULTADOS DE LA PRUEBA ---")
    if not cut_timestamps:
        print("No se detectaron cortes de escena.")
    else:
        print(f"Se detectaron {len(cut_timestamps)} cortes de escena en los siguientes segundos:")
        print([f"{ts:.2f}s" for ts in cut_timestamps])
    print("-------------------------------------")

if __name__ == "__main__":
    run_test()
