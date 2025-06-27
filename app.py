import gradio as gr
import traceback
import logging
import platform
import json
import os
import sys
import tempfile
import shutil

# --- Configuration Loading ---
def load_config():
    """Loads configuration from config.json, with safe defaults."""
    config_path = 'config.json'
    default_config = {
        'device': 'cpu',
        'analysis_level': 'Balanced (every 2 seconds)',
        'frames_per_clip': 8
    }
    if not os.path.exists(config_path):
        return default_config
    with open(config_path, 'r') as f:
        try:
            user_config = json.load(f)
            # Ensure all keys are present, falling back to default if missing
            return {**default_config, **user_config}
        except json.JSONDecodeError:
            return default_config

config = load_config()
DEVICE = config.get('device', 'cpu').lower()
ANALYSIS_LEVEL = config.get('analysis_level', 'Balanced (every 2 seconds)')
FRAMES_PER_CLIP = config.get('frames_per_clip', 8)

# --- Set JAX platform before imports ---
if DEVICE == 'cpu':
    os.environ['JAX_PLATFORMS'] = 'cpu'
elif 'JAX_PLATFORMS' in os.environ:
    del os.environ['JAX_PLATFORMS']

print(f"--- Running on device: {os.environ.get('JAX_PLATFORMS', 'gpu (default)')} ---")

# --- Add project root to sys.path and import project modules ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.video_processor import extract_clips, get_clip_thumbnails, get_video_duration, save_scenes
from src.embedding_extractor import extract_video_embeddings
from src.scene_detector import find_scene_cuts
from src.model_loader import load_videoprism_video_model
from src.utils import format_analysis_log, create_zip_from_scenes

# --- Global Variables ---
VIDEO_MODEL_FN = None

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler("app_errors.log"), logging.StreamHandler()])

# --- Model Loading ---
def get_video_model():
    global VIDEO_MODEL_FN
    if VIDEO_MODEL_FN is None:
        try:
            print("Loading video model...")
            use_fp16 = (DEVICE == 'gpu' and not (platform.system() == "Darwin" and platform.machine() == "arm64"))
            VIDEO_MODEL_FN = load_videoprism_video_model(model_name='base', use_fp16=use_fp16)
            print(f"Video model loaded. FP16: {use_fp16}")
        except Exception as e:
            logging.error(f"FATAL: Could not load video model: {e}\n{traceback.format_exc()}")
            raise
    return VIDEO_MODEL_FN

# --- Main Processing Logic ---
def process_video_in_one_go(video_path, threshold, progress=gr.Progress(track_tqdm=True)):
    logging.info("--- Initiating full video processing ---")
    scenes_dir = None
    try:
        if not video_path:
            return [], "Please upload a video.", "Analysis not started.", None, None, gr.update(visible=False)
        
        # Map user-friendly analysis level to clip duration in seconds
        level_to_duration = {
            'High Quality (every second)': 1,
            'Balanced (every 2 seconds)': 2,
            'Fast (every 4 seconds)': 4
        }
        clip_duration_secs = level_to_duration.get(ANALYSIS_LEVEL, 2)

        print(f"Analysis settings: Level='{ANALYSIS_LEVEL}', Frames/Clip={FRAMES_PER_CLIP}, Clip Duration={clip_duration_secs}s")
        print("Step 1: Analyzing video and extracting clips...")
        
        clips = extract_clips(video_path, clip_duration=clip_duration_secs, resolution=216, num_frames_per_clip=FRAMES_PER_CLIP)
        if not clips:
            return [], "Could not extract clips.", "Extraction failed.", None, None, gr.update(visible=False)

        thumbnail_images = get_clip_thumbnails(clips)
        model_fn = get_video_model()
        video_embeddings = extract_video_embeddings(clips, model_fn, batch_size=16)
        duration = get_video_duration(video_path)
        clip_duration = duration / len(clips) if clips else 0
        cut_timestamps, similarities, cut_indices = find_scene_cuts(video_path, video_embeddings, threshold, clip_duration)
        analysis_log = format_analysis_log(threshold, similarities, cut_timestamps)
        msg = f"Analysis complete. Found {len(cut_timestamps)} scene cuts."
        gallery_items = [(img, f"Scene Cut!" if i in set(cut_indices) else f"Clip {i}") for i, img in enumerate(thumbnail_images)]

        if not cut_timestamps:
            return gallery_items, msg, analysis_log, None, None, gr.update(visible=False)

        print("Step 2: Generating scene clips...")
        scenes_dir = tempfile.mkdtemp()
        scene_files = save_scenes(video_path, cut_timestamps, scenes_dir)

        if not scene_files:
            if scenes_dir: shutil.rmtree(scenes_dir)
            return gallery_items, msg, analysis_log, None, None, gr.update(visible=False)

        print("Step 3: Creating ZIP file...")
        zip_path = create_zip_from_scenes(scene_files)

        return gallery_items, msg, analysis_log, scene_files, zip_path, gr.update(value=zip_path, visible=True)

    except Exception as e:
        logging.error(f"Error in process_video_in_one_go: {e}\n{traceback.format_exc()}")
        if scenes_dir and os.path.exists(scenes_dir): shutil.rmtree(scenes_dir)
        raise gr.Error("An internal error occurred. Details logged to app_errors.log.")

# --- Settings Logic ---
def save_config_and_exit(device, analysis_level, frames_per_clip):
    new_config = {
        'device': 'cpu' if 'cpu' in device.lower() else 'gpu',
        'analysis_level': analysis_level,
        'frames_per_clip': int(frames_per_clip)
    }
    print(f"Saving configuration: {new_config}")
    with open('config.json', 'w') as f:
        json.dump(new_config, f, indent=2)
    print("Configuration saved. Exiting application. Please restart from the terminal.")
    os._exit(0)

# --- Gradio UI ---
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Tab("Video Processor"):
            # ... (Video Processor UI remains the same)
            gr.Markdown("# VideoPrism Scene Segmentation")
            gr.Markdown("Upload a video, analyze for scenes, then generate them for preview and download.")
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Input Video")
                    similarity_threshold = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.05, label="Similarity Threshold")
                    submit_btn = gr.Button("1. Analyze & Generate Scenes", variant="primary")
                    analysis_result_output = gr.Textbox(label="Analysis Result", interactive=False)
                    download_file = gr.File(label="Download Scenes ZIP", interactive=False, visible=False)
                with gr.Column(scale=2):
                    scene_gallery = gr.Gallery(label="Generated Scenes (Preview)", columns=4, height="auto")
                    gallery = gr.Gallery(label="Analyzed Clips Thumbnails", show_label=True, columns=8, height="auto")
                    analysis_log_output = gr.Textbox(label="Detailed Analysis Log", lines=15, interactive=False)
            submit_btn.click(fn=process_video_in_one_go, inputs=[video_input, similarity_threshold], outputs=[gallery, analysis_result_output, analysis_log_output, scene_gallery, download_file, download_file])

        with gr.Tab("Settings"):
            with gr.Column():
                gr.Markdown("## Performance & Quality Settings")
                gr.Markdown("Adjust these settings to balance speed and accuracy. **A manual restart is required for changes to take effect.**")
                
                device_selector = gr.Radio(["CPU (Slower, Less Power)", "GPU (Faster, More Power)"], value="CPU (Slower, Less Power)" if DEVICE == 'cpu' else "GPU (Faster, More Power)", label="Processing Device")
                
                analysis_level_selector = gr.Radio(
                    ['High Quality (every second)', 'Balanced (every 2 seconds)', 'Fast (every 4 seconds)'],
                    value=ANALYSIS_LEVEL,
                    label="Analysis Detail Level",
                    info="'High' is more accurate but slower. 'Fast' is much quicker but may miss short scenes."
                )
                
                frames_per_clip_slider = gr.Slider(
                    minimum=4, maximum=16, step=4, value=FRAMES_PER_CLIP, label="Frames to Analyze per Clip",
                    info="Fewer frames are faster but give the AI less context to judge the scene."
                )

                save_btn = gr.Button("Save and Exit")
                save_btn.click(fn=save_config_and_exit, inputs=[device_selector, analysis_level_selector, frames_per_clip_slider], outputs=[])
    return demo

if __name__ == "__main__":
    app_ui = create_ui()
    app_ui.launch(server_name='0.0.0.0', share=True)
