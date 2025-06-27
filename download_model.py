# download_model.py
# This script's only purpose is to download the VideoPrism model and cache it locally.
# Run this once from the command line to ensure the model is available for the main app.

import os
import sys

# Add the project root to the sys.path to find the src modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.model_loader import load_videoprism_video_model

def main():
    """Triggers the model download and caching mechanism."""
    print("--- INICIANDO DESCARGA DEL MODELO ---")
    print("Esto puede tardar varios minutos, pero solo se hará una vez.")
    print("Por favor, espera a que el proceso termine...")
    
    try:
        # Calling this function will handle the download and caching.
        # Calling this function will now use the default fp16=False for compatibility
        load_videoprism_video_model(model_name='base')
        print("\n--- ¡DESCARGA COMPLETA! ---")
        print("El modelo ha sido guardado en el caché local.")
        print("Ahora puedes iniciar la aplicación principal ('app.py') sin problemas.")
    except Exception as e:
        print(f"\n--- ERROR DURANTE LA DESCARGA ---")
        print(f"Ocurrió un error: {e}")
        print("Por favor, revisa tu conexión a internet e inténtalo de nuevo.")

if __name__ == "__main__":
    main()
