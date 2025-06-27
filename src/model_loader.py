import jax
import jax.numpy as jnp
import videoprism.models as vp
from sentence_transformers import SentenceTransformer

# Mapeo de nombres cortos/comunes a los nombres oficiales de la librería
MODEL_NAME_MAP = {
    'b': 'videoprism_public_v1_base',
    'base': 'videoprism_public_v1_base',
    'l': 'videoprism_public_v1_large',
    'large': 'videoprism_public_v1_large',
    'videoprism_public_v1_base': 'videoprism_public_v1_base',
    'videoprism_public_v1_large': 'videoprism_public_v1_large',
}

def load_videoprism_video_model(model_name: str, use_fp16: bool = False):
    """Carga el modelo de video VideoPrism y devuelve una función JIT para la inferencia."""
    
    official_model_name = MODEL_NAME_MAP.get(model_name)
    if not official_model_name:
        raise ValueError(f"Nombre de modelo no válido: '{model_name}'. Modelos disponibles: {list(MODEL_NAME_MAP.keys())}")

    print(f"Cargando modelo de video VideoPrism ({official_model_name}, fp16: {use_fp16})...")
    
    flax_model = vp.MODELS[official_model_name]()
    loaded_state = vp.load_pretrained_weights(official_model_name)

    if use_fp16:
        loaded_state = jax.tree_util.tree_map(
            lambda x: x.astype(jax.numpy.bfloat16) if x.dtype == jax.numpy.float32 else x,
            loaded_state
        )

    @jax.jit
    def video_model_fn(video_tensor):
        # NOTA: Los prints dentro de una función JIT solo se ejecutan en la primera compilación.
        return flax_model.apply(loaded_state, video_tensor, train=False)[0]

    # Calentamiento del modelo para compilarlo
    print("Calentando y compilando el modelo de video JIT...")
    # Usamos una resolución compatible (múltiplo de 18) y el tipo de dato correcto
    dummy_input_dtype = jax.numpy.bfloat16 if use_fp16 else jax.numpy.float32
    dummy_input = jax.numpy.zeros((1, 16, 234, 234, 3), dtype=dummy_input_dtype)
    _ = video_model_fn(dummy_input)
    print("Modelo de video listo.")

    return video_model_fn

def load_text_model(model_name: str = 'all-MiniLM-L6-v2'):
    """
    Carga un modelo de sentence-transformers para la extracción de embeddings de texto.
    """
    model = SentenceTransformer(model_name)
    return model
