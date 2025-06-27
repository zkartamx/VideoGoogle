import os
import zipfile
import numpy as np

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

def create_zip_from_scenes(scene_files):
    """Creates a ZIP file from a list of scene file paths."""
    if not scene_files:
        return None
    
    # The zip will be created in the same directory as the scenes.
    scenes_dir = os.path.dirname(scene_files[0])
    zip_path = os.path.join(scenes_dir, "scenes.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for scene_file in scene_files:
            zipf.write(scene_file, os.path.basename(scene_file))
    
    return zip_path
