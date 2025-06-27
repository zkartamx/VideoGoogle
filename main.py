import argparse
import os
import sys
import cv2
import numpy as np

# Añadir la ruta del proyecto al sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.model_loader import load_videoprism_video_model
from src.video_processor import extract_clips, save_clip
from src.embedding_extractor import extract_video_embeddings
from src.scene_detector import find_scene_cuts

def main(args):
    print("--- INICIANDO PRUEBA DE CONSOLA ---")
    
    # --- Carga de Modelo ---
    print("Cargando modelo de video...")
    video_model_fn = load_videoprism_video_model(model_name='base', use_fp16=True)
    print("Modelo de video cargado.")

    # --- Procesamiento de Video ---
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {args.video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / fps
    cap.release()
    print(f"Video Info: Duración={total_duration:.2f}s, FPS={fps:.2f}")

    print("\n1. Extrayendo clips...")
    clips = extract_clips(args.video_path, resolution=234, num_frames_per_clip=16)
    if not clips:
        print("Error: No se pudieron extraer clips.")
        return
    print(f"   => Clips extraídos: {len(clips)}")
    clip_duration = total_duration / len(clips)

    # --- Extracción de Embeddings ---
    print("\n2. Extrayendo embeddings...")
    video_embeddings = extract_video_embeddings(clips, video_model_fn, batch_size=32)
    print(f"   => Shape de embeddings: {video_embeddings.shape}")

    # --- Detección de Cortes ---
    print(f"\n3. Detectando cortes con umbral = {args.threshold}...")
    cut_timestamps, similarities = find_scene_cuts(video_embeddings, args.threshold, clip_duration)
    
    print(f"   => Similitudes calculadas: {len(similarities)}")
    if similarities:
        print(f"   => Similitud (min, max, avg): ({min(similarities):.3f}, {max(similarities):.3f}, {np.mean(similarities):.3f})")
    
    if not cut_timestamps:
        print("\n--- RESULTADO: No se detectaron cortes de escena. ---")
    else:
        print(f"\n--- RESULTADO: Se detectaron {len(cut_timestamps)} cortes de escena en los siguientes segundos: ---")
        print([f"{ts:.2f}s" for ts in cut_timestamps])

    print("\n--- PRUEBA DE CONSOLA FINALIZADA ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realiza una prueba de detección de escenas desde la consola.")
    parser.add_argument("video_path", type=str, help="Ruta al archivo de video.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Umbral de similitud para la detección de cortes.")
    
    parsed_args = parser.parse_args()
    main(parsed_args)

