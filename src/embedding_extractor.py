import numpy as np
from tqdm import tqdm

def extract_video_embeddings(clips, video_model_fn, batch_size=32):
    """
    Extrae embeddings de una lista de clips de video usando el modelo VideoPrism.
    Devuelve un único array de NumPy con todos los embeddings.
    """
    if not clips:
        return np.array([])

    all_embeddings_list = []
    num_batches = int(np.ceil(len(clips) / batch_size))

    for i in tqdm(range(num_batches), desc="Extracting video embeddings"):
        batch_clips = clips[i * batch_size : (i + 1) * batch_size]
        
        if not batch_clips:
            continue
            
        batch_array = np.array(batch_clips)
        # El modelo devuelve embeddings por token: (batch, num_tokens, dim)
        token_embeddings = video_model_fn(batch_array)
        
        # Promediamos los tokens para obtener un embedding por clip
        clip_embeddings = np.array(token_embeddings).mean(axis=1)
        
        # Normalizamos el embedding del clip
        normalized_embeddings = clip_embeddings / np.linalg.norm(clip_embeddings, axis=-1, keepdims=True)
        
        all_embeddings_list.append(normalized_embeddings)
        
    return np.vstack(all_embeddings_list)

def extract_text_embedding(text, text_model):
    """
    Convierte un texto en un embedding usando un modelo SentenceTransformer.
    """
    # El método .encode() se encarga de la tokenización y devuelve el embedding final.
    text_embedding = text_model.encode(text)
    return text_embedding
