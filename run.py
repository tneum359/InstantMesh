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
from einops import rearrange
from tqdm import tqdm
from glob import glob
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from torchvision.utils import make_grid
import json
import random
import traceback
import subprocess # Import subprocess
import base64
from io import BytesIO

# --- Add parent directory to sys.path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PARENT_DIR)
print(f"--- DEBUG: Added {PARENT_DIR} to sys.path ---")

# --- Local utils ---
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground

# --- Helper: Composite RGBA over white ---
def rgba_to_rgb_white(img):
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        # Alpha composite needs both images to be RGBA
        return Image.alpha_composite(background.convert('RGBA'), img).convert('RGB') 
    else:
        return img.convert('RGB')

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
# --- End Helper Function ---

# --- Core Processing Function ---
def process_image(args, config, model_config, infer_config, device, 
                  pipeline, model, gemini_verifier, rembg_session, input_cameras,
                  input_image_path, intermediate_dir, output_dir, is_gemini_pass):
    """Processes a single image, optionally using Gemini."""
    img_name = os.path.splitext(os.path.basename(input_image_path))[0]
    num_candidates = args.num_candidates if is_gemini_pass else 1 # Use arg for Gemini pass, 1 otherwise
    use_gemini_this_pass = is_gemini_pass and (gemini_verifier is not None)
    
    print(f'\n[{img_name}] Processing {"with Gemini" if is_gemini_pass else "no Gemini"} (from {input_image_path}) ...')
    print(f"  Intermediate outputs -> {intermediate_dir}")
    print(f"  Final 3D object -> {output_dir}")

    # --- Define output paths for this specific input/pass ---
    base_obj_name = f'generation_{img_name}' # Temporary name before rename
    final_obj_name = 'obj_with_gemini.obj' if is_gemini_pass else 'obj_no_gemini.obj'
    output_obj_path = os.path.join(output_dir, base_obj_name) # Path before renaming
    final_output_obj_path = os.path.join(output_dir, final_obj_name)

    try:
        # --- Load input image ---
        try:
            input_image = Image.open(input_image_path).convert('RGBA') # Ensure RGBA for consistency
        except Exception as e:
            print(f"  Error opening image {input_image_path}: {e}. Skipping.")
            return

        # --- Preprocessing ---
        print("  Preprocessing input image...")
        if not args.no_rembg:
            if rembg_session is None:
                 print("  Warning: --no_rembg not set, but rembg_session is None. Skipping background removal.")
            else:
                 print("  Removing background...")
                 try:
                     input_image = remove_background(input_image, rembg_session)
                 except Exception as e:
                     print(f"  Error removing background: {e}. Proceeding with original image.")
            # input_image = resize_foreground(input_image, 0.85) # Optional

        # --- Candidate Generation Loop ---
        best_group_seed = -1
        best_group_score = -float('inf')
        best_group_images_tensor = None
        best_group_pil_grid = None
        best_group_pil_list_rgb = None # Store the final RGB list for reconstruction

        # Create candidate groups (even if only 1)
        num_actual_candidates = 1 if not is_gemini_pass else args.num_candidates
        print(f"  Generating and evaluating {num_actual_candidates} candidate group(s)...")
        
        for candidate_idx in range(num_actual_candidates):
            current_seed = random.randint(0, 2**32 - 1)
            print(f"    --- Candidate Group {candidate_idx + 1}/{num_actual_candidates} ---")
            print(f"    Generating multiview images with seed: {current_seed}... (Res: {args.gen_width}x{args.gen_height})") 
            seed_everything(current_seed)

            # Generate the grid of multiview images
            try:
                 output_image_pil_grid = pipeline( 
                     input_image, 
                     num_inference_steps=args.diffusion_steps,
                     width=args.gen_width, 
                     height=args.gen_height,
                     guidance_scale=3.0, # Default guidance scale
                 ).images[0]
            except Exception as e:
                 print(f"    Error during diffusion pipeline generation: {e}")
                 traceback.print_exc()
                 continue # Skip to next candidate if generation fails

            # ---> START: Background Removal for Generated Multiviews <--- 
            # Split the grid into individual PIL images
            w, h = output_image_pil_grid.size
            sub_w, sub_h = w // 2, h // 3 # Assuming 3 rows, 2 columns
            multiview_pil_list_raw = []
            for row in range(3):
                for col in range(2):
                    box = (col * sub_w, row * sub_h, (col + 1) * sub_w, (row + 1) * sub_h)
                    multiview_pil_list_raw.append(output_image_pil_grid.crop(box))

            multiview_pil_list_nobg_rgb = [] # Store final RGB images
            if not args.no_rembg and rembg_session is not None:
                print(f"    Removing background from {len(multiview_pil_list_raw)} generated views...")
                multiview_pil_list_nobg_rgba = []
                for i, img_pil in enumerate(multiview_pil_list_raw):
                     try:
                          # Apply rembg
                          img_nobg_rgba = rembg.remove(img_pil, session=rembg_session)
                          multiview_pil_list_nobg_rgba.append(img_nobg_rgba)
                     except Exception as rembg_err:
                          print(f"      Warning: rembg failed on view {i}: {rembg_err}. Using original view.")
                          # Fallback: use original if rembg fails
                          multiview_pil_list_nobg_rgba.append(img_pil.convert("RGBA")) 
                
                # Composite RGBA on white background
                for i, img_rgba in enumerate(multiview_pil_list_nobg_rgba):
                     try:
                          img_rgb_on_white = rgba_to_rgb_white(img_rgba)
                          multiview_pil_list_nobg_rgb.append(img_rgb_on_white)
                     except Exception as composite_err:
                          print(f"      Warning: Compositing failed on view {i}: {composite_err}. Using original view.")
                          # Fallback: use original converted to RGB if compositing fails
                          multiview_pil_list_nobg_rgb.append(multiview_pil_list_raw[i].convert("RGB"))
            else:
                # If rembg is disabled, just convert raw images to RGB
                print(f"    Background removal on generated views skipped.")
                for img_pil in multiview_pil_list_raw:
                     multiview_pil_list_nobg_rgb.append(img_pil.convert("RGB"))
            # ---> END: Background Removal <--- 

            # --- Prepare TENSOR for reconstruction model (use multiview_pil_list_nobg_rgb) ---
            # Apply transformations (ToTensor, Resize) to the background-removed RGB images
            transform = v2.Compose([
                v2.ToTensor(), # Converts PIL to Tensor [C, H, W] in [0, 1]
            ])
            try:
                images_tensor_list = [transform(img) for img in multiview_pil_list_nobg_rgb]
                images_tensor = torch.stack(images_tensor_list) # [6, C, H, W]
                images_tensor = images_tensor.unsqueeze(0).to(device) # [B, V, C, H, W], B=1, V=6
                # Resize for the reconstruction model input
                images_tensor = v2.functional.resize(images_tensor, 320, interpolation=3, antialias=True).clamp(0, 1) 
                print(f"    Processed images_tensor shape: {images_tensor.shape} dtype: {images_tensor.dtype} range: {images_tensor.min()} {images_tensor.max()}")
            except Exception as e:
                 print(f"    Error transforming processed PIL images to tensor: {e}")
                 traceback.print_exc()
                 continue # Skip candidate if tensor prep fails
            # --- End Tensor Prep ---

            current_score = 0.0
            candidate_metadata = None
            # Evaluate with Gemini only in the Gemini pass
            if is_gemini_pass and gemini_verifier is not None:
                # We need to pass the RGBA list before compositing for Gemini scoring
                # If rembg failed, we created RGBA versions as fallback
                images_pil_for_gemini = multiview_pil_list_nobg_rgba if (not args.no_rembg and rembg_session is not None) else [img.convert("RGBA") for img in multiview_pil_list_raw]
                
                # ---> Prepare inputs dictionary for Gemini score method <--- 
                print(f"    Preparing data for Gemini scoring...")
                gemini_input_data = {"images": [], "prompt": args.gemini_prompt} # Use prompt from args
                for img in images_pil_for_gemini:
                     try:
                          buffered = BytesIO()
                          img.save(buffered, format="PNG")
                          img_str = base64.b64encode(buffered.getvalue()).decode()
                          gemini_input_data["images"].append({
                              "mime_type": "image/png",
                              "data": img_str
                          })
                     except Exception as e:
                          print(f"      Warning: Failed to encode an image for Gemini: {e}")
                          # Optionally skip this image or the whole candidate?
                          # For now, let's allow it to proceed with fewer images if one fails
                # ---> End Prepare inputs <--- 

                print(f"    Scoring candidate group {candidate_idx + 1} with Gemini...")
                if not gemini_input_data["images"]:
                     print(f"    Skipping Gemini scoring: No images successfully encoded.")
                else:
                     try:
                          # Call score with the prepared dictionary
                          gemini_result = gemini_verifier.score(inputs=gemini_input_data) 
                          
                          # Process the result (check success, extract score)
                          if gemini_result.get("success"):
                               candidate_metadata = gemini_result.get("result")
                               if candidate_metadata and isinstance(candidate_metadata.get('overall_score'), (int, float)):
                                    current_score = candidate_metadata['overall_score']
                                    print(f"    Gemini Score: {current_score:.4f}")
                               else:
                                    print("    Warning: Gemini did not return a valid overall_score.")
                          else:
                               print(f"    Warning: Gemini evaluation failed: {gemini_result.get('error', 'Unknown error')}")
                               if gemini_result.get('raw_response'):
                                    print(f"      Raw Gemini Response: {gemini_result['raw_response'][:200]}...") # Print first 200 chars

                     except Exception as e:
                          print(f"    Error calling Gemini score method: {e}")
                          traceback.print_exc()
                          # Keep score 0, proceed without Gemini result for this candidate
            else:
                 # If not Gemini pass, score is 0, use first candidate
                 pass

            # Save intermediate images for this candidate *only* in Gemini pass
            if is_gemini_pass:
                candidate_subdir = os.path.join(intermediate_dir, f'candidate_{candidate_idx+1}_seed_{current_seed}')
                os.makedirs(candidate_subdir, exist_ok=True)
                output_image_pil_grid.save(os.path.join(candidate_subdir, 'multiview_grid_raw.png'))
                # Save the background-removed RGB versions used for reconstruction
                for i, img in enumerate(multiview_pil_list_nobg_rgb):
                     img.save(os.path.join(candidate_subdir, f'view_{i}_nobg_rgb.png'))
                if candidate_metadata:
                    meta_path = os.path.join(candidate_subdir, 'gemini_scores.json')
                    try:
                        with open(meta_path, 'w') as f:
                            json.dump(candidate_metadata, f, indent=4)
                    except Exception as e:
                        print(f"  Warning: Failed to save Gemini metadata: {e}")

            # Check if this candidate is the best so far
            if current_score > best_group_score:
                print(f"    New best group found (Score: {current_score:.4f}) Seed: {current_seed}")
                best_group_score = current_score
                best_group_seed = current_seed
                best_group_images_tensor = images_tensor # Use the processed tensor
                best_group_pil_grid = output_image_pil_grid # Save the raw grid for reference
                best_group_pil_list_rgb = multiview_pil_list_nobg_rgb # Save the final list used
        # --- End Candidate Loop ---

        if best_group_images_tensor is None:
             print("  Error: No successful candidate group generated.")
             raise RuntimeError("Failed to generate any valid candidate group.")

        print(f"  Selected best group for {img_name} (Seed: {best_group_seed}, Score: {best_group_score:.4f})")

        # --- Save Intermediate Outputs for the Best Group --- 
        # Save the chosen best raw grid image (optional, maybe comment out if not needed)
        # best_grid_raw_path = os.path.join(intermediate_dir, f'best_multiview_grid_raw_seed_{best_group_seed}.png')
        # if best_group_pil_grid:
        #      best_group_pil_grid.save(best_grid_raw_path)
        #      print(f"  Saved best candidate raw grid to {best_grid_raw_path}")

        # Save the chosen best processed RGB list to the main intermediate dir
        if best_group_pil_list_rgb:
             best_list_dir = os.path.join(intermediate_dir, f'best_views_nobg_rgb_seed_{best_group_seed}')
             os.makedirs(best_list_dir, exist_ok=True)
             for i, img in enumerate(best_group_pil_list_rgb):
                  img.save(os.path.join(best_list_dir, f'view_{i}.png'))
             print(f"  Saved best candidate processed views to {best_list_dir}")

             # --- Create and Save Processed Grid --- 
             try:
                  # Convert the processed PIL list back to tensors to use make_grid
                  transform_to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) # Use ToImage for PIL
                  tensors_for_grid = [transform_to_tensor(img) for img in best_group_pil_list_rgb]
                  if tensors_for_grid:
                       processed_grid_tensor = make_grid(tensors_for_grid, nrow=2) # Arrange in 2 columns
                       processed_grid_pil = F.to_pil_image(processed_grid_tensor)
                       processed_grid_path = os.path.join(intermediate_dir, f'best_multiview_grid_processed_seed_{best_group_seed}.png')
                       processed_grid_pil.save(processed_grid_path)
                       print(f"  Saved best candidate processed grid to {processed_grid_path}")
                  else:
                       print("  Warning: No processed images available to create processed grid.")
             except Exception as e:
                  print(f"  Warning: Failed to create or save processed grid: {e}")
                  traceback.print_exc()
             # --- End Processed Grid Saving ---

        # --- Reconstruction Step (using best_group_images_tensor) ---
        print("  Starting reconstruction...")
        # Use the selected best tensor
        images = best_group_images_tensor 

        # --- Camera Selection ---
        # Create cameras ONCE outside the loop, select here
        base_input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale)
        current_input_cameras = base_input_cameras # Start with all 6
        if args.view == 4:
            print("  Selecting 4 views for reconstruction...")
            indices = torch.tensor([0, 2, 4, 5]).long() # Standard 4 views
            images = images[:, indices]
            current_input_cameras = base_input_cameras[:, indices] # Select corresponding cameras
        current_input_cameras = current_input_cameras.to(device) # Move selected cameras to device

        # --- Mesh Extraction ---
        with torch.no_grad():
            print("  Generating triplanes...")
            # Ensure model and cameras are on the correct device
            planes = model.to(device).forward_planes(images, current_input_cameras)
            print("  Extracting mesh...")
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
        print(f"\n !!! UNHANDLED ERROR processing {img_name} ({'Gemini pass' if is_gemini_pass else 'No Gemini pass'}) !!!")
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
    parser.add_argument('--num_candidates', type=int, default=1, help='Number of candidate groups for Gemini scoring (if > 1).')
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
    # (Keep this logic, assuming checkpoints might be relative or downloaded)
    # ... (Load Diffusion Pipeline) ...
    pipeline = None
    try:
        print('Loading diffusion model ...')
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2", 
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )
        print('Loading custom white-background unet ...')
        # Try finding UNet relative to script parent first, then config path, then download
        unet_path_rel = os.path.join(PARENT_DIR, "ckpts", "diffusion_pytorch_model.bin") 
        if os.path.exists(infer_config.unet_path):
             unet_ckpt_path = infer_config.unet_path # Allow override from config
             print(f"  Using UNet from config: {unet_ckpt_path}")
        elif os.path.exists(unet_path_rel):
             unet_ckpt_path = unet_path_rel
             print(f"  Using local UNet: {unet_ckpt_path}")
        else:
            print(f"Custom UNet not found locally, downloading...")
            unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
        
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        pipeline.unet.load_state_dict(state_dict, strict=True)
        pipeline = pipeline.to(device)
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
                print(f"Error initializing Gemini Verifier: {e}. Gemini scoring disabled.")
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
        model = model.to(device)
        if IS_FLEXICUBES:
            # Check if geometry needs initialization
            if hasattr(model, 'init_flexicubes_geometry'): 
                 model.init_flexicubes_geometry(device, fovy=30.0)
            else:
                 print("Warning: Model does not have init_flexicubes_geometry method.")
        model = model.eval()
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
