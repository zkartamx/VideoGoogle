import numpy as np

def find_text_matches(video_embeddings: list, text_embedding: np.ndarray, threshold: float, clip_duration: float = 1.0) -> list:
    """
    Encuentra y agrupa clips de video cuya similitud con el embedding de texto supera un umbral.

    Retorna:
        Una lista de tuplas, donde cada tupla contiene el timestamp de inicio y fin de una escena coincidente.
    """
    if not video_embeddings or text_embedding is None:
        return []

    # Normalizar todos los embeddings para un cálculo de similitud coseno preciso
    video_embs = np.stack(video_embeddings)
    video_embs_norm = video_embs / np.linalg.norm(video_embs, axis=1, keepdims=True)
    text_emb_norm = text_embedding / np.linalg.norm(text_embedding)

    # Calcular la similitud coseno entre el texto y todos los clips de video
    similarities = np.dot(video_embs_norm, text_emb_norm.T).flatten()
    
    print("\nBuscando clips que coincidan con el texto...")
    for i, sim in enumerate(similarities):
        print(f"  - Clip {i}: similitud = {sim:.3f}")

    # Encontrar todos los clips que superan el umbral de similitud
    matching_indices = np.where(similarities > threshold)[0]

    if not len(matching_indices):
        return []

    # Agrupar clips consecutivos en escenas continuas
    scenes = []
    start_index = matching_indices[0]
    for i in range(1, len(matching_indices)):
        # Si el clip actual no es consecutivo al anterior, cerramos la escena actual
        if matching_indices[i] != matching_indices[i-1] + 1:
            end_index = matching_indices[i-1]
            scenes.append((start_index * clip_duration, (end_index + 1) * clip_duration))
            start_index = matching_indices[i]
    
    # Añadir la última escena detectada
    end_index = matching_indices[-1]
    scenes.append((start_index * clip_duration, (end_index + 1) * clip_duration))
    
    return scenes
