import os
import sys
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
import torchvision.transforms.functional as F
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from glob import glob
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from torchvision.utils import make_grid
import json
import random
import traceback
import subprocess
import base64
from io import BytesIO
import shutil

# Add parent directory to Python path to find verifiers module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verifiers.gemini_verifier import GeminiVerifier

# --- Add InstantMesh directory to sys.path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PARENT_DIR)
sys.path.append(SCRIPT_DIR)
print(f"--- DEBUG: Added {PARENT_DIR} and {SCRIPT_DIR} to sys.path ---")

# Try different import paths
try:
    # Try direct import first (using symlink)
    from utils.train_util import instantiate_from_config
    from utils.camera_util import (
        FOV_to_intrinsics, 
        get_zero123plus_input_cameras,
        get_circular_camera_poses,
    )
    from utils.mesh_util import save_obj, save_obj_with_mtl
    from utils.infer_util import remove_background, resize_foreground, save_video
    print("Successfully imported from utils (direct)")
except ImportError:
    try:
        # Try with src prefix
        from src.utils.train_util import instantiate_from_config
        from src.utils.camera_util import (
            FOV_to_intrinsics, 
            get_zero123plus_input_cameras,
            get_circular_camera_poses,
        )
        from src.utils.mesh_util import save_obj, save_obj_with_mtl
        from src.utils.infer_util import remove_background, resize_foreground, save_video
        print("Successfully imported from src.utils")
    except ImportError as e:
        print(f"Failed to import required modules. Error: {e}")
        print(f"Current sys.path: {sys.path}")
        print(f"Contents of {SCRIPT_DIR}:")
        try:
            print(os.listdir(SCRIPT_DIR))
            print("\nContents of src/utils:")
            print(os.listdir(os.path.join(SCRIPT_DIR, 'src', 'utils')))
        except Exception as e:
            print(f"Could not list directories: {e}")
        raise

# --- Local utils ---
try:
    from InstantMesh.src.utils.train_util import instantiate_from_config
    from InstantMesh.src.utils.camera_util import (
        FOV_to_intrinsics, 
        get_zero123plus_input_cameras,
        get_circular_camera_poses,
    )
    from InstantMesh.src.utils.mesh_util import save_obj, save_obj_with_mtl
    from InstantMesh.src.utils.infer_util import remove_background, resize_foreground, save_video
except ImportError as e:
    print(f"Failed to import from InstantMesh.src.utils: {e}")
    raise

# --- Helper: Composite RGBA over white ---
def rgba_to_rgb_white(img):
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        # Alpha composite needs both images to be RGBA
        return Image.alpha_composite(background.convert('RGBA'), img).convert('RGB') 
    else:
        return img.convert('RGB')

# --- Helper: Create image grid ---
def create_image_grid(images, rows=3, cols=2):
    """Create a grid of images.
    
    Args:
        images: List of numpy arrays (images)
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        
    Returns:
        numpy array containing the grid image
    """
    if not images:
        return None
        
    # Get dimensions of first image
    h, w = images[0].shape[:2]
    
    # Create empty grid
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    # Fill grid with images
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        i, j = idx // cols, idx % cols
        grid[i*h:(i+1)*h, j*w:(j+1)*w] = img[:, :, :3]  # Ensure RGB
        
    return grid

# --- Helper: Camera function (assuming it's needed globally or passed) ---
def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras

# --- Function to safely convert mesh component to NumPy ---
def safe_to_numpy(component):
    if isinstance(component, memoryview):
        return np.asarray(component)
    elif hasattr(component, 'data'):
        # Check if the .data attribute itself is a memoryview
        if isinstance(component.data, memoryview):
            return np.asarray(component.data)
        # Otherwise, assume .data can be moved to cpu/converted
        try:
            return component.data.cpu().numpy()
        except AttributeError:
             print(f"  Warning: component.data has no .cpu() or .numpy() method. Trying direct conversion.")
             try:
                  return np.array(component.data)
             except Exception as e:
                  print(f"  Error converting component.data to numpy: {e}. Returning None.")
                  return None
    elif isinstance(component, np.ndarray):
        return component # Already numpy
    elif isinstance(component, torch.Tensor):
        # Handle direct tensors without a .data attribute if needed
        try:
             return component.cpu().numpy()
        except AttributeError:
             print(f"  Warning: Tensor component has no .cpu() or .numpy() method. Trying direct conversion.")
             try:
                  return np.array(component)
             except Exception as e:
                  print(f"  Error converting Tensor component to numpy: {e}. Returning None.")
                  return None
    else:
        # Fallback for other types
        try:
            return np.array(component)
        except Exception as e:
             print(f"  Warning: Cannot convert component of type {type(component)} to numpy: {e}. Returning None.")
             return None

# --- Core Processing Function ---
def process_image(args, config, model_config, infer_config, device, 
                  pipeline, model, gemini_verifier, rembg_session, input_cameras,
                  input_image_path, intermediate_dir, output_dir, is_gemini_pass):
    """Processes a single image, optionally using Gemini."""
    img_name_base = os.path.splitext(os.path.basename(input_image_path))[0]
    num_candidates = args.num_candidates if is_gemini_pass else 1 # Use arg for Gemini pass, 1 otherwise
    use_gemini_this_pass = is_gemini_pass and (gemini_verifier is not None)
    
    print(f'\n[{img_name_base}] Processing {"with Gemini" if is_gemini_pass else "no Gemini"} (from {input_image_path}) ...')
    print(f"  Intermediate outputs -> {intermediate_dir}")
    print(f"  Final 3D object -> {output_dir}")

    # --- Clear intermediate directory if it's a Gemini pass and dir exists ---
    if is_gemini_pass and os.path.exists(intermediate_dir):
        print(f"  Clearing previous contents of intermediate directory: {intermediate_dir}")
        for item_name in os.listdir(intermediate_dir):
            item_path = os.path.join(intermediate_dir, item_name)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path) # Use shutil.rmtree for directories
            except Exception as e:
                print(f'    Failed to delete {item_path}. Reason: {e}')
    # Ensure the intermediate_dir exists after potential cleaning (or for the first run)
    os.makedirs(intermediate_dir, exist_ok=True)

    # --- Define output paths for this specific input/pass ---
    base_obj_name = f'generation_{img_name_base}' # Temporary name before rename
    final_obj_name = 'obj_with_gemini.obj' if is_gemini_pass else 'obj_no_gemini.obj'
    output_obj_path = os.path.join(output_dir, base_obj_name) # Path before renaming
    final_output_obj_path = os.path.join(output_dir, final_obj_name)

    try:
        # --- Load input image & Preprocessing ---
        try:
            # Always load as RGBA first to handle transparency consistently
            initial_input_pil = Image.open(input_image_path).convert('RGBA') 
        except Exception as e:
            print(f"  Error opening image {input_image_path}: {e}. Skipping this image.")
            return # Exit process_image if initial load fails

        print("  Preprocessing input image...")
        input_image_pil_nobg = initial_input_pil # Default to original RGBA if rembg fails or is skipped
        if not args.no_rembg and rembg_session is not None:
            print("  Removing background from input image...")
            try:
                # remove_background should return an RGBA image if successful
                removed_bg_img = remove_background(initial_input_pil, rembg_session) 
                if removed_bg_img:
                    input_image_pil_nobg = removed_bg_img
                else:
                    print("  Warning: Background removal returned None. Using original image.")
            except Exception as e:
                print(f"  Error removing background from input: {e}. Proceeding with original RGBA image.")
        
        # For the diffusion pipeline, convert the (potentially background-removed) image to RGB
        input_image_for_pipeline = input_image_pil_nobg.convert('RGB') 

        # --- Candidate Generation Loop ---
        best_candidate_score = 0.0
        best_candidate_images = None
        best_candidate_seed = None
        best_candidate_metadata = None
        candidate_count = 0

        # Continue generating candidates until we either:
        # 1. Get a score above target_score, or
        # 2. Reach max_candidates, or
        # 3. Have tried at least min_candidates
        while (candidate_count < args.min_candidates or 
               (best_candidate_score < args.target_score and candidate_count < args.num_candidates)):
            
            candidate_count += 1
            print(f"\n    --- Candidate Group {candidate_count}/{args.num_candidates} ---")
            
            # Generate a new random seed for this candidate
            candidate_seed = random.randint(0, 2**32 - 1)
            print(f"    Generating multiview images with seed: {candidate_seed}...")
            
            # Set the seed for this candidate
            torch.manual_seed(candidate_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(candidate_seed)
            np.random.seed(candidate_seed)
            random.seed(candidate_seed)

            # Generate multiview images
            # Convert input image to tensor and ensure correct device/dtype
            # input_tensor = torch.from_numpy(np.array(input_image_for_pipeline)).permute(2, 0, 1).float() / 255.0
            # input_tensor = input_tensor.unsqueeze(0).to(device=device, dtype=torch.float16)
            
            with torch.cuda.amp.autocast(): # type: ignore
                output_image = pipeline(
                    input_image_for_pipeline, # Pass the PIL image directly
                    num_inference_steps=args.diffusion_steps,
                ).images[0]

            # Save the grid image
            output_image.save(os.path.join(intermediate_dir, f'candidate_{candidate_count}_seed_{candidate_seed}.png'))

            # Split the grid into individual views
            w, h = output_image.size
            sub_w, sub_h = w // 2, h // 3
            images_pil = []
            for row in range(3):
                for col in range(2):
                    box = (col * sub_w, row * sub_h, (col + 1) * sub_w, (row + 1) * sub_h)
                    images_pil.append(output_image.crop(box))

            # Score with Gemini if available
            if gemini_verifier is not None:
                print("    Preparing data for Gemini scoring...")
                
                # Convert original input image to base64
                original_img_bytes = BytesIO()
                input_image_pil_nobg.save(original_img_bytes, format='PNG')
                original_img_b64 = base64.b64encode(original_img_bytes.getvalue()).decode('utf-8')
                
                # Convert candidate images to base64
                candidate_views_b64 = []
                for img in images_pil:
                    img_bytes = BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                    candidate_views_b64.append(img_b64)
                
                # Create input for Gemini
                gemini_api_input_payload = {
                    "original_image_b64": original_img_b64,
                    "candidate_views_b64": candidate_views_b64
                }
                
                print("    Scoring candidate group with Gemini...")
                gemini_result = gemini_verifier.score(inputs=gemini_api_input_payload)
                
                if gemini_result["success"]:
                    candidate_score = gemini_result["result"]["overall_score"]
                    print(f"    Gemini Score: {candidate_score:.4f}")
                    
                    # Update best candidate if this one is better
                    if candidate_score > best_candidate_score:
                        best_candidate_score = candidate_score
                        best_candidate_seed = candidate_seed
                        best_candidate_metadata = {
                            "seed": candidate_seed,
                            "score": candidate_score,
                            "gemini_scores": gemini_result["result"]
                        }
                        best_candidate_images = images_pil
                        print(f"    New best group found (Score: {candidate_score:.4f}) Seed: {candidate_seed}")
                else:
                    print(f"    Warning: Gemini evaluation failed for candidate {candidate_count}: {gemini_result['error']}")
                    # If Gemini fails, use this candidate as best if we don't have one yet
                    if best_candidate_metadata is None:
                        best_candidate_metadata = {
                            "seed": candidate_seed,
                            "score": 0.0,
                            "gemini_scores": None
                        }
                        best_candidate_images = images_pil
                        best_candidate_seed = candidate_seed
                        print(f"    Using candidate as best (no score) Seed: {candidate_seed}")

            # Print progress towards target score
            if gemini_verifier is not None:
                print(f"    Current best score: {best_candidate_score:.4f} (Target: {args.target_score})")
                if best_candidate_score >= args.target_score:
                    print(f"    Target score achieved! Stopping candidate generation.")
                elif candidate_count >= args.num_candidates:
                    print(f"    Reached maximum number of candidates ({args.num_candidates}).")
                elif candidate_count < args.min_candidates:
                    print(f"    Continuing to meet minimum candidate requirement ({args.min_candidates}).")

        # After the loop, save the best candidate's outputs
        if best_candidate_metadata is not None:
            print(f"  Selected best group (Seed: {best_candidate_seed}, Score: {best_candidate_score:.4f})")
            
            # Save best candidate's views
            for i, img in enumerate(best_candidate_images):
                img.save(os.path.join(intermediate_dir, f'best_view_{i}_seed_{best_candidate_seed}.png'))
            
            # Save best candidate's metadata
            with open(os.path.join(intermediate_dir, "best_candidate_metadata.json"), "w") as f:
                json.dump(best_candidate_metadata, f, indent=2)
            
            # Save best candidate's Gemini scores if available
            if best_candidate_metadata.get("gemini_scores"):
                with open(os.path.join(intermediate_dir, "best_candidate_gemini_scores.json"), "w") as f:
                    json.dump(best_candidate_metadata["gemini_scores"], f, indent=2)

            # Convert best candidate images to tensor format for reconstruction
            images = []
            for img in best_candidate_images:
                img_np = np.asarray(img, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().float()
                images.append(img_tensor)
            images = torch.stack(images)

            # --- Reconstruction Step (using best_group_images_tensor) ---
            print("  Starting reconstruction...")
            print("  Generating triplanes...")
            
            # Clear CUDA cache before reconstruction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Resize images to a smaller size for reconstruction
            target_size = (512, 512)  # Reduced from 2048x2048
            resized_images = []
            for img in images:
                resized_img = img.unsqueeze(0).permute(0, 2, 3, 1).resize(target_size, Image.Resampling.LANCZOS)
                resized_images.append(resized_img)
            
            # Convert list of PIL images to tensor
            images_tensor = torch.stack(resized_images).permute(0, 3, 1, 2).to(device)
            
            # Create cameras for reconstruction
            base_input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale)
            current_input_cameras = base_input_cameras
            if args.view == 4:
                print("  Selecting 4 views for reconstruction...")
                indices = torch.tensor([0, 2, 4, 5]).long()  # Standard 4 views
                images_tensor = images_tensor[indices]
                current_input_cameras = base_input_cameras[:, indices]  # Select corresponding cameras
            current_input_cameras = current_input_cameras.to(device)
            
            # Enable gradient checkpointing if available
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'gradient_checkpointing_enable'):
                model.encoder.gradient_checkpointing_enable()
            
            # Generate triplanes with memory optimization
            with torch.cuda.amp.autocast():  # Use automatic mixed precision
                planes = model.to(device).forward_planes(images_tensor, current_input_cameras)
            
            # Clear memory after triplane generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Generate mesh
            print("  Generating mesh...")
            mesh_out = model.extract_mesh(
                planes,
                use_texture_map=args.export_texmap,
                **infer_config,
            )
            print(f"  Saving mesh to {output_obj_path} (temp name)... ")
            if args.export_texmap:
                # Check if mesh_out has the expected components
                if len(mesh_out) == 5:
                    vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                    # Apply safe_to_numpy to each component needed for save_obj_with_mtl
                    verts_np = safe_to_numpy(vertices)
                    uvs_np = safe_to_numpy(uvs) 
                    faces_np = safe_to_numpy(faces)
                    mesh_tex_idx_np = safe_to_numpy(mesh_tex_idx)
                    tex_map_np = safe_to_numpy(tex_map.permute(1, 2, 0)) # Handle permute if tex_map is tensor
                    if verts_np is None or uvs_np is None or faces_np is None or mesh_tex_idx_np is None or tex_map_np is None:
                        print(f"  Error: Failed to convert one or more mesh components to NumPy. Cannot save OBJ.")
                    else:
                        save_obj_with_mtl(verts_np, uvs_np, faces_np, mesh_tex_idx_np, tex_map_np, output_obj_path)
                else:
                    print("  Warning: export_texmap requested but mesh output format unexpected. Saving with vertex colors.")
                    vertices, faces, vertex_colors = mesh_out # Fallback assuming vertex colors
                    # --- Use Safe Conversion Function --- 
                    verts_np = safe_to_numpy(vertices)
                    faces_np = safe_to_numpy(faces)
                    colors_np = safe_to_numpy(vertex_colors)

                    if verts_np is None or faces_np is None or colors_np is None:
                        print(f"  Error: Failed to convert one or more mesh components to NumPy. Cannot save OBJ.")
                    else:
                        # Handle potential shape mismatch if color conversion failed differently
                        if colors_np.shape[0] != verts_np.shape[0]:
                            print(f"  Warning: Vertex and color array lengths mismatch ({verts_np.shape[0]} vs {colors_np.shape[0]}). Using fallback gray.")
                            colors_np = np.ones_like(verts_np) * 0.5
                        save_obj(verts_np, faces_np, colors_np, output_obj_path)
            else:
                # Ensure mesh_out has vertex colors
                if len(mesh_out) == 3:
                    vertices, faces, vertex_colors = mesh_out
                    # --- Use Safe Conversion Function --- 
                    verts_np = safe_to_numpy(vertices)
                    faces_np = safe_to_numpy(faces)
                    
                    if verts_np is None or faces_np is None:
                        print(f"  Error: Failed to convert vertices or faces to NumPy. Cannot save OBJ.")
                    else:
                        colors_np = safe_to_numpy(vertex_colors)
                        if colors_np is None:
                            print(f"  Warning: Failed to convert vertex_colors to NumPy. Using fallback gray.")
                            colors_np = np.ones_like(verts_np) * 0.5
                        save_obj(verts_np, faces_np, colors_np, output_obj_path)
                else:
                    print("  Warning: Vertex colors expected but mesh output format unexpected. Saving without colors.")
                    # Handle cases where only vertices and faces might be returned
                    if len(mesh_out) == 2:
                        vertices, faces = mesh_out
                        # --- Use Safe Conversion Function --- 
                        verts_np = safe_to_numpy(vertices)
                        faces_np = safe_to_numpy(faces)
                        
                        if verts_np is None or faces_np is None:
                            print(f"  Error: Failed to convert vertices or faces to NumPy. Cannot save OBJ.")
                        else:
                            dummy_colors = np.ones_like(verts_np) * 0.5 # Gray
                            save_obj(verts_np, faces_np, dummy_colors, output_obj_path)
                    else:
                        print(f"  Error: Unexpected number of items ({len(mesh_out)}) returned by extract_mesh. Cannot save OBJ.")

            # --- Rename the saved OBJ ---
            if os.path.exists(output_obj_path):
                try:
                    # Ensure target path doesn't exist (optional, os.rename might overwrite)
                    if os.path.exists(final_output_obj_path):
                        print(f"  Warning: Overwriting existing file {final_output_obj_path}")
                        os.remove(final_output_obj_path)
                    os.rename(output_obj_path, final_output_obj_path)
                    print(f"  Mesh saved to {final_output_obj_path}")
                except Exception as e:
                    print(f"  Error renaming {output_obj_path} to {final_output_obj_path}: {e}")
            else:
                print(f"  Warning: Mesh file {output_obj_path} not found after saving.")

    except Exception as e:
        print(f"\n !!! UNHANDLED ERROR processing {img_name_base} ({'Gemini pass' if is_gemini_pass else 'No Gemini pass'}) !!!")
        print(f"Error: {e}")
        traceback.print_exc()

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input image or directory.')
    parser.add_argument('--output_intermediate_path', type=str, default='outputs/intermediate_images', help='Base directory for intermediate outputs.')
    parser.add_argument('--output_3d_path', type=str, default='outputs/output_3d', help='Base directory for final 3D outputs.')
    parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
    parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
    parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of views for reconstruction.')
    parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
    parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
    parser.add_argument('--save_video', action='store_true', help='Save a circular-view video (Not fully supported in batch mode).')
    parser.add_argument('--num_candidates', type=int, default=8, help='Maximum number of candidates to generate.')
    parser.add_argument('--min_candidates', type=int, default=3, help='Minimum number of candidates to generate.')
    parser.add_argument('--target_score', type=float, default=8.0, help='Target score to achieve.')
    parser.add_argument('--gemini_prompt', type=str, default=None, help='Prompt for Gemini verifier.')
    parser.add_argument('--gen_width', type=int, default=640, help='Width for multiview generation.')
    parser.add_argument('--gen_height', type=int, default=960, help='Height for multiview generation.')
    parser.add_argument('--batch_mode', action='store_true', help='Process all PNGs in input_path directory.')
    args = parser.parse_args()

    # --- Basic Setup ---
    print("--- Initializing Models and Environment ---")
    seed_everything(args.seed)
    try:
        config = OmegaConf.load(args.config)
        config_name = os.path.basename(args.config).replace('.yaml', '')
        model_config = config.model_config
        infer_config = config.infer_config
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        exit(1)

    IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Final Path Checks and Directory Creation (Simplified) ---
    print(f"Using Input Path: {args.input_path}")
    # Use the robust check relying on subprocess ls as a fallback
    is_input_dir_os = os.path.isdir(args.input_path)
    is_input_file = os.path.isfile(args.input_path)
    can_subprocess_ls = False
    if not is_input_dir_os and not is_input_file: # Only check subprocess if basic checks fail
        print(f"DEBUG: os.path.isdir/isfile failed for {args.input_path}. Attempting subprocess ls...")
        try:
            result = subprocess.run(['ls', args.input_path], capture_output=True, text=True, check=False, timeout=15)
            if result.returncode == 0:
                 can_subprocess_ls = True
                 print(f"DEBUG: Subprocess ls succeeded. Treating as directory.")
            else:
                 print(f"DEBUG: Subprocess ls failed (Code: {result.returncode}). stderr: {result.stderr}")
        except Exception as e:
             print(f"DEBUG: Error during subprocess ls check: {e}")

    is_effectively_input_dir = is_input_dir_os or can_subprocess_ls

    if args.batch_mode and not is_effectively_input_dir:
        print(f"Error: Batch mode requires --input_path ('{args.input_path}') to be an accessible directory.")
        exit(1)
    
    input_file_to_process_single = None
    if not args.batch_mode:
        if is_input_file:
            input_file_to_process_single = args.input_path
        elif is_effectively_input_dir: 
            print(f"Input path '{args.input_path}' is directory (single mode), finding first PNG...")
            try:
                 input_files = sorted(glob(os.path.join(args.input_path, '*.png')))
                 if not input_files:
                      print(f"Error: No PNG images found in directory '{args.input_path}'.")
                      exit(1)
                 input_file_to_process_single = input_files[0]
                 print(f"  Processing first image: {input_file_to_process_single}")
            except Exception as e:
                 print(f"Error reading directory '{args.input_path}': {e}")
                 exit(1)
        else:
            print(f"Error: Input path '{args.input_path}' is not a valid file or accessible directory.")
            exit(1)

    # Ensure base output dirs exist using provided/default args
    try:
         os.makedirs(args.output_intermediate_path, exist_ok=True)
         os.makedirs(args.output_3d_path, exist_ok=True)
         print(f"Intermediate output base set to: {args.output_intermediate_path}")
         print(f"3D output base set to: {args.output_3d_path}")
    except Exception as e:
         print(f"Error creating output directories: {e}")
         exit(1)

    # --- Load Gemini Prompt (Assume relative path works or user provides full path) ---
    DEFAULT_GEMINI_PROMPT_FALLBACK = "Evaluate multiview images for 3D reconstruction."
    if args.gemini_prompt is None:
        # Use path relative to this script's parent (repo root)
        prompt_file_path = os.path.join(PARENT_DIR, "verifiers", "verifier_prompt.txt") 
        print(f"Attempting to load Gemini prompt from {prompt_file_path}")
        try:
            with open(prompt_file_path, 'r') as f:
                args.gemini_prompt = f.read()
            print("  Successfully loaded default prompt from file.")
        except Exception as e:
            print(f"  Warning: Error reading prompt file ({e}). Using fallback.")
            args.gemini_prompt = DEFAULT_GEMINI_PROMPT_FALLBACK
    else:
        print("Using provided --gemini_prompt.")

    # --- Load models (Diffusion, UNet, Recon) --- 
    pipeline = None
    try:
        print('Loading diffusion model ...')
        # Load the pipeline, hint float16 dtype
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2", 
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            # variant="fp16", # Removed: Variant doesn't exist
            torch_dtype=torch.float16, 
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )

        print('Loading custom white-background unet ...')
        # Try finding UNet relative to script parent first, then config path, then download
        unet_path_rel = os.path.join(PARENT_DIR, "ckpts", "diffusion_pytorch_model.bin") 
        if os.path.exists(infer_config.unet_path):
            unet_ckpt_path = infer_config.unet_path 
            print(f"  Using UNet from config: {unet_ckpt_path}")
        elif os.path.exists(unet_path_rel):
            unet_ckpt_path = unet_path_rel
            print(f"  Using local UNet: {unet_ckpt_path}")
        else:
            print(f"Custom UNet not found locally, downloading...")
            unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
        
        # Load custom UNet weights onto the (likely) CPU/float32 UNet
        state_dict = torch.load(unet_ckpt_path, map_location='cpu') 
        pipeline.unet.load_state_dict(state_dict, strict=True) 
        print("Custom UNet weights loaded.")

        # Move the entire pipeline to the target device first.
        pipeline = pipeline.to(device)
        print("Pipeline moved to device.")

        # Explicitly set components to half and force parameters
        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
            pipeline.unet.half() # Convert module to half
            for param in pipeline.unet.parameters():
                param.data = param.data.to(device).half() # Force params
            print("UNet set to half and parameters forced.")

        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            pipeline.vae.half() # Convert module to half
            for param in pipeline.vae.parameters():
               param.data = param.data.to(device).half() # Force params
            print("VAE set to half and parameters forced.")

        if hasattr(pipeline, 'vision_encoder') and pipeline.vision_encoder is not None:
            pipeline.vision_encoder.half() # Convert module to half
            for param in pipeline.vision_encoder.parameters():
               param.data = param.data.to(device).half() # Force params
            print("Vision Encoder set to half and parameters forced.")
        
        print("Pipeline components processed for device and dtype.")

        # Enable memory optimizations for diffusion pipeline
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, 'enable_vae_slicing'):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, 'enable_model_cpu_offload'):
            pipeline.enable_model_cpu_offload()
            
        print("Diffusion pipeline loaded.")
    except Exception as e:
        print(f"Error loading diffusion model: {e}")
        traceback.print_exc()
        exit(1)

    # ... (Initialize Gemini Verifier) ... 
    gemini_verifier = None
    gemini_available_flag = False
    if args.num_candidates > 1:
        if os.getenv("GEMINI_API_KEY"):
            try:
                # Ensure verifiers package is importable
                from verifiers.gemini_verifier import GeminiVerifier 
                print("Initializing Gemini Verifier...")
                gemini_verifier = GeminiVerifier(gemini_prompt=args.gemini_prompt)
                print("Gemini Verifier initialized successfully.")
                gemini_available_flag = True
            except ImportError:
                 print("Warning: verifiers.gemini_verifier not found. Gemini scoring disabled.")
            except Exception as e:
                print(f"Error initializing Gemini Verifier: {e}. Proceeding without Gemini scoring.")
        else:
            print("Warning: GEMINI_API_KEY not found. Gemini scoring disabled.")
    else:
        print("Gemini scoring disabled (num_candidates=1).")

    # ... (Load Reconstruction Model) ...
    model = None
    try:
        print('Loading reconstruction model ...')
        model = instantiate_from_config(model_config)
        model_ckpt_filename = f"{config_name.replace('-', '_')}.ckpt"
        # Try finding model relative to script parent first, then config path, then download
        model_path_rel = os.path.join(PARENT_DIR, "ckpts", model_ckpt_filename) 
        if hasattr(infer_config, 'model_path') and os.path.exists(infer_config.model_path):
            model_ckpt_path = infer_config.model_path # Allow override
            print(f"  Using Recon Model from config: {model_ckpt_path}")
        elif os.path.exists(model_path_rel):
            model_ckpt_path = model_path_rel
            print(f"  Using local Recon Model: {model_ckpt_path}")
        else:
            print(f"Reconstruction model not found locally, downloading...")
            model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=model_ckpt_filename, repo_type="model")
        
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'renderer' not in k}
        model.load_state_dict(state_dict, strict=False)
        
        # Enable memory optimizations for reconstruction model
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'gradient_checkpointing_enable'):
            model.encoder.gradient_checkpointing_enable()

        model = model.to(device)
        if IS_FLEXICUBES:
            # Check if geometry needs initialization
            if hasattr(model, 'init_flexicubes_geometry'): 
                model.init_flexicubes_geometry(device, fovy=30.0)
            else:
                print("Warning: Model does not have init_flexicubes_geometry method.")
        model = model.eval()

        # Clear CUDA cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        print("Reconstruction model loaded.")
    except Exception as e:
        print(f"Error loading reconstruction model: {e}")
        traceback.print_exc()
        exit(1)

    # ... (Define Rembg Session) ...
    rembg_session = None
    if not args.no_rembg:
        try:
            rembg_session = rembg.new_session()
            print("Rembg session created.")
        except Exception as e:
            print(f"Warning: Failed to create rembg session: {e}. Background removal disabled.")
            args.no_rembg = True 

    # --- Batch or Single Image Processing ---
    if args.batch_mode:
        # --- Add Warning if Gemini isn't configured for batch mode ---
        if not gemini_available_flag:
            print("\nWarning: Batch mode is selected, but Gemini scoring is not available or enabled.")
            print("         (Requires --num_candidates > 1 AND GEMINI_API_KEY to be set).")
            print("         No processing will be performed in this run.")
            # Optionally exit here, or let it proceed (currently proceeds and skips each image)
            # exit(1)

        # --- Begin Batch Processing ---
        input_dir = args.input_path
        all_images = sorted(glob(os.path.join(input_dir, '*.png')))
        if not all_images:
             print(f"Error: No PNG images found in batch input directory via glob: {input_dir}.")
             exit(1)
        print(f"--- Starting Batch Mode (Gemini Pass Only): Found {len(all_images)} PNG images in {input_dir} ---")

        for img_path in tqdm(all_images, desc="Processing Batch (Gemini Pass)"):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            intermediate_subdir = os.path.join(args.output_intermediate_path, f'data_{img_name}')
            output_subdir = os.path.join(args.output_3d_path, f'output_{img_name}')
            os.makedirs(intermediate_subdir, exist_ok=True)
            os.makedirs(output_subdir, exist_ok=True)

            # --- Removed Pass 1 (No Gemini) ---
            # process_image(args, config, model_config, infer_config, device,
            #               pipeline, model, gemini_verifier, rembg_session, None, # Pass None for cameras
            #               img_path, intermediate_subdir, output_subdir, is_gemini_pass=False)

            # --- Run Only Pass 2 (With Gemini, if available) ---
            if gemini_available_flag:
                 print(f"\n[{img_name}] Processing Gemini Pass (from {img_path}) ...")
                 try:
                      process_image(args, config, model_config, infer_config, device,
                                    pipeline, model, gemini_verifier, rembg_session, None, # Pass None for cameras
                                    img_path, intermediate_subdir, output_subdir, is_gemini_pass=True)
                 except Exception as e:
                      print(f"\n !!! UNHANDLED ERROR processing {img_name} (Gemini pass) !!!")
                      print(f"Error: {e}")
                      traceback.print_exc()
                      print(f"  Skipping to next image due to error.")
            else:
                 # This message will now appear if the warning at the start wasn't triggered
                 # (e.g., if API key exists but num_candidates=1 was somehow forced - less likely now)
                 # Or if the loop continued despite the initial warning.
                 print(f"  [{img_name}] Skipping: Gemini pass required but not available/enabled for this run.")

        print("--- Batch processing complete. ---")

    else: # Single Image Mode
        # Use the file determined earlier
        input_file_to_process = input_file_to_process_single 
        if input_file_to_process is None:
             print("Error: Could not determine single input file to process.")
             exit(1)
             
        # Setup paths for single image mode
        img_name = os.path.splitext(os.path.basename(input_file_to_process))[0]
        intermediate_subdir = os.path.join(args.output_intermediate_path, f'data_{img_name}') 
        output_subdir = os.path.join(args.output_3d_path, f'output_{img_name}') 
        os.makedirs(intermediate_subdir, exist_ok=True)
        os.makedirs(output_subdir, exist_ok=True)

        should_run_gemini_single = args.num_candidates > 1 and gemini_available_flag
        
        process_image(args, config, model_config, infer_config, device, 
                      pipeline, model, gemini_verifier, rembg_session, None, # Pass None for cameras
                      input_file_to_process, intermediate_subdir, output_subdir, 
                      is_gemini_pass=should_run_gemini_single)

        print("--- Single image processing complete. ---")

print("\n--- Script Finished ---")
