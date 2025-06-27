# Segmentador de Escenas con Google VideoPrism

Este proyecto divide un video en clips de escenas distintas usando embeddings semánticos extraídos con el modelo VideoPrism de Google.

## Requisitos
- macOS, Python 3.11
- FFmpeg instalado en el sistema (brew install ffmpeg)
- **Nota**: Este proyecto requiere `tensorflow-macos` debido a una dependencia interna de la librería `videoprism`. Esto se ha incluido en `requirements.txt`.

## Instalación
```bash
# En la raíz del proyecto:
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Instalar videoprism sin sus dependencias (para evitar conflicto con tensorflow)
pip install --no-deps git+https://github.com/google-deepmind/videoprism.git
```

## Modelo VideoPrism
El modelo **google/videoprism** sólo está disponible en JAX/Flax. No utilice `transformers` o `torch`.
1. Se clona e instala el paquete desde GitHub via pip (ya está en `requirements.txt`).
2. Use `src/model_loader.py` para inicializar y obtener la función de inferencia.

## Uso de la herramienta

1.  Coloca el video que deseas procesar en la carpeta `input_videos/`.
2.  Ejecuta el siguiente comando en tu terminal:

```bash
# Reemplaza 'tu_video.mp4' con el nombre de tu archivo
python main.py --input input_videos/tu_video.mp4
```

Los clips resultantes se guardarán automáticamente en la carpeta `output_scenes/`.

## Estructura del Proyecto
```
VideoGoogle/
├── input_videos/     # <--- Coloca tus videos aquí
├── output_scenes/    # <--- Aquí se guardan los resultados
├── src/                # <--- Código fuente del programa
├── venv/               # <--- Entorno virtual de Python
├── main.py             # <--- Script principal para ejecutar
├── requirements.txt
└── README.md
```

## Estructura del Código
- **src/model_loader.py**: carga el modelo VideoPrism en JAX/Flax.
- **src/video_processor.py**: lee video y extrae clips preprocesados.
- **src/embedding_extractor.py**: obtiene embeddings de cada clip.
- **src/scene_detector.py**: detecta cortes de escena según similitud.
- **main.py**: orquestador CLI que ejecuta el flujo completo.
