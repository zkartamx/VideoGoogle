import gradio as gr
import numpy as np
import cv2
import os
import sys
import tempfile
import zipfile
import shutil
import logging
import traceback

# Configure logging to write to a file
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app_errors.log',
    filemode='a'  # Append to the log file, create if it doesn't exist
)

# Add the project root to the sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.model_loader import load_videoprism_video_model
from src.video_processor import extract_clips, get_clip_thumbnails, get_video_duration, save_scenes
from src.embedding_extractor import extract_video_embeddings
from src.scene_detector import find_scene_cuts

# --- Model Loading (Lazy Loading) ---
VIDEO_MODEL_FN = None

def get_video_model():
    """Loads the model on the first call and caches it globally."""
    global VIDEO_MODEL_FN
    if VIDEO_MODEL_FN is None:
        try:
            print("Loading video model for the first time...")
            # Dynamically set FP16 based on GPU availability for better portability
            import platform
            # For now, disable FP16 on Apple Silicon as it can be unstable with jax-metal
            USE_FP16 = not (platform.system() == "Darwin" and platform.machine() == "arm64")
            VIDEO_MODEL_FN = load_videoprism_video_model(model_name='base', use_fp16=USE_FP16)
            print(f"Video model loaded and cached. Using FP16: {USE_FP16}")
        except Exception as e:
            error_details = traceback.format_exc()
            logging.error(f"FATAL: Could not load video model: {e}\n{error_details}")
            # Re-raise the exception to halt the app, as it cannot function without the model.
            raise
    return VIDEO_MODEL_FN

def format_analysis_log(threshold, similarities, cut_timestamps):
    """Creates a formatted text log of the analysis."""
    log_lines = [f"Detecting cuts with threshold = {threshold}...\n"]
    if similarities:
        for i, sim in enumerate(similarities):
            log_lines.append(f"Clip {i}→{i+1}: sim = {sim:.3f}")

    log_lines.append("\n--- STATISTICS ---")
    if similarities:
        log_lines.append(f"Similarities calculated: {len(similarities)}")
        log_lines.append(f"Similarity (min, max, avg): ({min(similarities):.3f}, {max(similarities):.3f}, {np.mean(similarities):.3f})")
    else:
        log_lines.append("No similarities calculated.")

    log_lines.append("\n--- RESULT ---")
    if not cut_timestamps:
        log_lines.append("No scene cuts were detected.")
    else:
        log_lines.append(f"Detected {len(cut_timestamps)} scene cuts at the following seconds:")
        log_lines.append(str([f"{ts:.2f}s" for ts in cut_timestamps]))

    return "\n".join(log_lines)

def process_video_in_one_go(video_path, threshold):
    """Analyzes video, generates scenes, and returns all UI updates in a single function."""
    logging.info("--- Initiating full video processing ---")
    try:
        if not video_path:
            return [], "Please upload a video.", "Analysis has not started.", None, None

        # 1. Analyze Video
        print("Step 1: Analyzing video and extracting clips...")
        clips = extract_clips(video_path, resolution=288, num_frames_per_clip=16)
        if not clips:
            return [], "Could not extract clips.", "Extraction failed.", None, None

        thumbnail_images = get_clip_thumbnails(clips)
        model_fn = get_video_model()
        video_embeddings = extract_video_embeddings(clips, model_fn, batch_size=32)
        duration = get_video_duration(video_path)
        clip_duration = duration / len(clips) if clips else 0
        cut_timestamps, similarities, cut_indices = find_scene_cuts(video_path, video_embeddings, threshold, clip_duration)
        analysis_log = format_analysis_log(threshold, similarities, cut_timestamps)
        msg = f"Analysis complete. Found {len(cut_timestamps)} scene cuts."
        gallery_items = [(img, f"Scene Cut!" if i in set(cut_indices) else f"Clip {i}") for i, img in enumerate(thumbnail_images)]

        # If no cuts, stop here and return analysis results
        if not cut_timestamps:
            print("No scene cuts detected. Skipping scene generation.")
            return gallery_items, msg, analysis_log, None, None

        # 2. Generate Scenes
        print(f"Step 2: Generating {len(cut_timestamps)} scenes...")
        scenes_dir = tempfile.mkdtemp()
        scene_paths = save_scenes(video_path, cut_timestamps, scenes_dir)
        if not scene_paths:
            shutil.rmtree(scenes_dir)
            return gallery_items, msg, analysis_log, None, None

        print("Step 3: Creating ZIP archive...")
        zip_path = os.path.join(scenes_dir, f"scenes_{os.path.basename(video_path)}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for scene_file in scene_paths:
                zipf.write(scene_file, os.path.basename(scene_file))
        
        print("Processing complete.")
        return gallery_items, msg, analysis_log, scene_paths, zip_path

    except Exception as e:
        error_details = traceback.format_exc()
        logging.error(f"Error in process_video_in_one_go: {e}\n{error_details}")
        if 'scenes_dir' in locals() and os.path.exists(scenes_dir):
            shutil.rmtree(scenes_dir)
        raise gr.Error("An internal error occurred. Details logged to app_errors.log.")

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## ✂️ VideoPrism Scene Segmentation")
    gr.Markdown("Upload a video to automatically analyze it, generate scene clips, and provide a downloadable ZIP file.")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Input Video")
            threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.01, label="Similarity Threshold")
            submit_btn = gr.Button("Analyze & Generate Scenes", variant="primary")
            info_output = gr.Textbox(label="Analysis Result", lines=3)
            download_file = gr.File(label="Download Scenes ZIP", interactive=False)

        with gr.Column(scale=2):
            scene_gallery = gr.Gallery(label="Generated Scenes (Preview)", columns=4, height="auto")
            gallery = gr.Gallery(label="Analyzed Clips Thumbnails", show_label=True, columns=8, height="auto")
            analysis_log_output = gr.Textbox(label="Detailed Analysis Log", lines=15, interactive=False)

    submit_btn.click(
        fn=process_video_in_one_go,
        inputs=[video_input, threshold_slider],
        outputs=[gallery, info_output, analysis_log_output, scene_gallery, download_file]
    )

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
