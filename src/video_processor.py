import os
import cv2
import numpy as np
import ffmpeg
import subprocess

def extract_clips(
    video_path: str,
    clip_duration: float = 1.0,
    resolution: int = 224,
    num_frames_per_clip: int = 8,
):
    """Divide un video en clips y los preprocesa."""
    target_size = (resolution, resolution)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_clip = int(fps * clip_duration)
    clips = []
    buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.append(frame)
        if len(buffer) >= frames_per_clip:
            clips.append(_preprocess_clip(buffer, target_size, num_frames_per_clip))
            buffer = []

    if buffer:
        clips.append(_preprocess_clip(buffer, target_size, num_frames_per_clip))

    cap.release()
    return clips

def _preprocess_clip(
    frames: list,
    target_size: tuple,
    sample_frames: int,
) -> np.ndarray:
    """Preprocesa y muestrea los frames de un clip."""
    proc = []
    for f in frames:
        f = cv2.resize(f, target_size)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        proc.append(f.astype(np.float32) / 255.0)

    if len(proc) > sample_frames:
        indices = np.linspace(0, len(proc) - 1, sample_frames, dtype=int)
        proc = [proc[i] for i in indices]
    elif len(proc) < sample_frames:
        # Duplicar frames si hay menos de los necesarios
        indices = np.round(np.linspace(0, len(proc) - 1, sample_frames)).astype(int)
        proc = [proc[i] for i in indices]

    return np.stack(proc, axis=0)

def save_clip(input_path: str, start_time: float, end_time: float, output_path: str):
    """Guarda un segmento de video usando ffmpeg."""
    command = [
        'ffmpeg',
        '-i', input_path,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-c', 'copy',
        '-y', # Sobrescribir archivo de salida si existe
        output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_clip_thumbnails(clips):
    """Extrae el frame central de cada clip como una imagen en memoria (NumPy array)."""
    thumbnails = []
    for i, clip in enumerate(clips):
        # Tomar el frame central
        central_frame = clip[len(clip) // 2]
        
        # Convertir de float32 [0,1] a uint8 [0,255]
        # Gradio espera imágenes en formato RGB, así que no se necesita conversión a BGR.
        img_uint8_rgb = (central_frame * 255).astype(np.uint8)
        thumbnails.append(img_uint8_rgb)
        
    return thumbnails
    return thumbnail_paths

def get_video_duration(video_path: str) -> float:
    """Obtiene la duración de un video en segundos."""
    try:
        probe = ffmpeg.probe(video_path)
        return float(probe['format']['duration'])
    except ffmpeg.Error as e:
        print(f"Error al obtener la duración del video: {e.stderr.decode('utf-8') if e.stderr else 'Unknown error'}")
        return 0.0

def save_scenes(input_video_path, cut_timestamps, output_dir):
    """
    Corta el video en escenas basadas en los timestamps y las guarda en un directorio.
    Utiliza un método de re-codificación para garantizar la creación fiable de clips cortos.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_duration = get_video_duration(input_video_path)
    
    # Definir siempre los límites de la escena para incluir todos los segmentos.
    scene_boundaries = sorted(list(set([0] + cut_timestamps + [video_duration])))
    
    output_paths = []
    for i in range(len(scene_boundaries) - 1):
        start_time = scene_boundaries[i]
        end_time = scene_boundaries[i+1]

        scene_path = os.path.join(output_dir, f"escena_{i+1:03d}.mp4")
        
        # Comando FFMPEG robusto: re-codifica en lugar de copiar.
        # Esto es más lento pero mucho más fiable para cortes precisos y clips cortos.
        command = [
            'ffmpeg',
            '-i', input_video_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-y',          # Sobrescribe el archivo de salida si existe
            scene_path
        ]
        
        try:
            # Se usa un timeout para evitar que ffmpeg se quede colgado en casos extraños.
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            output_paths.append(scene_path)
        except subprocess.CalledProcessError as e:
            print(f"Error al cortar la escena {i+1} (ffmpeg falló): {e}")
        except subprocess.TimeoutExpired:
            print(f"Error al cortar la escena {i+1}: ffmpeg tardó demasiado.")

    return output_paths
