import numpy as np
import cv2

def _calculate_frame_diff(frame1, frame2):
    """Calcula la diferencia de histograma entre dos fotogramas para detectar cortes."""
    # Reducir la escala para mejorar el rendimiento
    frame1_small = cv2.resize(frame1, (64, 36))
    frame2_small = cv2.resize(frame2, (64, 36))
    
    # Calcular histogramas para cada canal de color
    hist1 = cv2.calcHist([frame1_small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([frame2_small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # Normalizar los histogramas para una comparación consistente
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # Comparar los histogramas (la distancia Chi-Cuadrado es buena para esto)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

def _refine_cut_location(video_path, coarse_cut_timestamp, search_radius_secs=1.5):
    """
    Refina un timestamp de corte aproximado buscando la máxima diferencia entre fotogramas
    en una pequeña ventana de tiempo alrededor del corte.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return coarse_cut_timestamp

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        return coarse_cut_timestamp

    # Definir la ventana de búsqueda en segundos y convertirla a fotogramas
    start_search_time = max(0, coarse_cut_timestamp - search_radius_secs)
    end_search_time = coarse_cut_timestamp + search_radius_secs
    start_frame = int(start_search_time * fps)
    end_frame = int(end_search_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    max_diff = -1
    refined_timestamp = coarse_cut_timestamp
    
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return coarse_cut_timestamp

    current_frame_num = start_frame + 1
    while current_frame_num <= end_frame:
        ret, current_frame = cap.read()
        if not ret:
            break
        
        diff = _calculate_frame_diff(prev_frame, current_frame)
        
        if diff > max_diff:
            max_diff = diff
            # El corte ocurre después del fotograma anterior, por lo que el timestamp es el del fotograma actual
            refined_timestamp = current_frame_num / fps
            
        prev_frame = current_frame
        current_frame_num += 1

    cap.release()
    return refined_timestamp

def find_scene_cuts(video_path, video_embeddings, threshold, clip_duration):
    """
    Encuentra los cortes de escena utilizando un enfoque de dos pasadas:
    1. Detección aproximada utilizando embeddings de clips (IA).
    2. Refinamiento preciso utilizando la diferencia a nivel de fotograma (Visión por Computadora).
    """
    # --- Paso 1: Detección Aproximada ---
    similarities = []
    for i in range(len(video_embeddings) - 1):
        emb1 = video_embeddings[i]
        emb2 = video_embeddings[i+1]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarities.append(similarity)

    coarse_cut_indices = [i for i, sim in enumerate(similarities) if sim < threshold]
    
    if not coarse_cut_indices:
        return [], similarities, []

    # --- Paso 2: Refinamiento de cada corte aproximado ---
    refined_timestamps = []
    for i in coarse_cut_indices:
        # El corte aproximado está en el límite entre el clip i y el i+1
        coarse_timestamp = (i + 1) * clip_duration
        refined_ts = _refine_cut_location(video_path, coarse_timestamp)
        refined_timestamps.append(refined_ts)

    # --- Post-procesamiento: Fusionar cortes que estén demasiado juntos ---
    if not refined_timestamps:
        return [], similarities, []
        
    merged_timestamps = [refined_timestamps[0]]
    for ts in refined_timestamps[1:]:
        # Fusionar si los cortes están a menos de 1 segundo de distancia para evitar duplicados
        if ts - merged_timestamps[-1] > 1.0:
            merged_timestamps.append(ts)
            
    # Recalcular los índices de los clips para la visualización en la galería
    final_cut_indices = [int(ts / clip_duration) for ts in merged_timestamps]

    return merged_timestamps, similarities, final_cut_indices
