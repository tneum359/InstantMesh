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
        best_group_data = {
            "avg_score": -float('inf'), "images_pil": None, "images_tensor": None,
            "gemini_scores": None, "seed": -1
        }
        
        print(f"  Generating and evaluating {num_candidates} candidate group(s)...")
        for i in range(num_candidates):
            print(f"    --- Candidate Group {i+1}/{num_candidates} ---")
            current_seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=device).manual_seed(current_seed)
            output_image_pil_grid = None 

            print(f"    Generating multiview images with seed: {current_seed}... (Res: {args.gen_width}x{args.gen_height}) ")
            try:
                # Generate the grid image
                output_image_pil_grid = pipeline( 
                    input_image.convert('RGB'), # Pipeline expects RGB
                    num_inference_steps=args.diffusion_steps,
                    generator=generator,
                    width=args.gen_width, 
                    height=args.gen_height,
                ).images[0]
            except Exception as e:
                print(f"    Error during image generation for group {i+1}: {e}")
                continue # Try next candidate

            if output_image_pil_grid is None:
                print(f"    Skipping candidate {i+1} due to generation failure.")
                continue
            
            # --- Process Generated Views ---
            try:
                # Split the grid into 6 separate views
                w, h = output_image_pil_grid.size
                if w == 0 or h == 0: raise ValueError("Generated grid image has zero dimension")
                single_w, single_h = w // 2, h // 3
                if single_w == 0 or single_h == 0: raise ValueError("Calculated single view dimension is zero")
                
                images_pil_list = [
                    output_image_pil_grid.crop((col * single_w, row * single_h, (col + 1) * single_w, (row + 1) * single_h))
                    for row in range(3) for col in range(2)
                ]
                
                # Remove background from each generated view (result is RGBA)
                if rembg_session:
                     images_pil_list_rgba = [remove_background(img, rembg_session=rembg_session) for img in images_pil_list]
                else: # If no rembg, assume generated has background we want to keep (or convert to white later)
                     images_pil_list_rgba = [img.convert('RGBA') for img in images_pil_list]

                # Convert cleaned PIL images (RGBA) to RGB on white background, then to tensor
                to_tensor = v2.ToTensor()
                images_rgb_list = [rgba_to_rgb_white(img) for img in images_pil_list_rgba]
                images_tensor = torch.stack([to_tensor(img) for img in images_rgb_list]) 
                images_tensor = images_tensor.unsqueeze(0) # Add batch dim -> (1, 6, 3, H, W)
                print("    Processed images_tensor shape:", images_tensor.shape, "dtype:", images_tensor.dtype, "range:", images_tensor.min().item(), images_tensor.max().item())
            
            except Exception as e:
                 print(f"    Error processing generated views for group {i+1}: {e}")
                 traceback.print_exc()
                 continue # Try next candidate

            # --- Gemini Evaluation (if applicable) ---
            avg_group_score = i # Default score if no Gemini
            gemini_result_data = None
            if use_gemini_this_pass:
                print("    Applying Gemini Verifier...")
                try:
                    # Gemini needs RGBA or RGB? Prepare inputs expects list of PIL
                    # Pass the RGBA list for evaluation if Gemini handles it, otherwise RGB list
                    gemini_inputs = gemini_verifier.prepare_inputs(
                        images=images_pil_list_rgba, # Pass RGBA list for evaluation
                        prompts=[args.gemini_prompt] * 6
                    )
                    gemini_result = gemini_verifier.score(inputs=gemini_inputs)
                    if gemini_result["success"]:
                        gemini_result_data = gemini_result["result"]
                        avg_group_score = gemini_result_data.get("overall_score", -1) # Handle missing score
                        print(f"    Gemini Score: {avg_group_score:.2f}")
                    else:
                        print(f"    Gemini Eval Error: {gemini_result.get('error', 'Unknown')}")
                        avg_group_score = -1 
                        gemini_result_data = None
                except Exception as e:
                    print(f"    Gemini Verification Error: {e}")
                    traceback.print_exc()
                    avg_group_score = -1
                    gemini_result_data = None

            # --- Update Best Group ---
            score_to_compare = avg_group_score
            # Update if score is better, OR if it's the first candidate in a no-Gemini pass
            if (score_to_compare > best_group_data["avg_score"]) or (num_candidates == 1 and i == 0): 
                print(f"    New best group found (Score: {score_to_compare:.4f}) Seed: {current_seed}")
                best_group_data["avg_score"] = score_to_compare
                best_group_data["images_pil"] = images_pil_list_rgba # Store the list of RGBA PIL images
                best_group_data["images_tensor"] = images_tensor # Store the (1, 6, 3, H, W) RGB tensor
                best_group_data["gemini_scores"] = gemini_result_data
                best_group_data["seed"] = current_seed
        # --- End Candidate Loop ---

        if best_group_data["images_tensor"] is None:
             print(f"  Failed to generate ANY valid candidates for {img_name}. Skipping reconstruction.")
             return # Skip reconstruction for this image

        print(f"  Selected best group for {img_name} (Seed: {best_group_data['seed']}, Score: {best_group_data['avg_score']:.4f})")

        # --- Save Intermediate Outputs (only if Gemini pass and data exists) ---
        if is_gemini_pass and best_group_data["images_pil"]:
            intermediate_image_path = os.path.join(intermediate_dir, f'intermediate_{img_name}.png')
            gemini_txt_path = os.path.join(intermediate_dir, f'gemini_output_{img_name}.txt')
            
            # Save intermediate grid image (composite on white)
            try:
                # Use the stored RGBA images and convert to RGB on white for saving grid
                img_tensors_rgb = [v2.ToTensor()(rgba_to_rgb_white(img)) for img in best_group_data["images_pil"]]
                if not img_tensors_rgb: raise ValueError("No images to create grid")
                grid = make_grid(torch.stack(img_tensors_rgb), nrow=2)
                grid_img = F.to_pil_image(grid)
                grid_img.save(intermediate_image_path)
                print(f"  Saved intermediate grid to {intermediate_image_path}")
            except Exception as e:
                print(f"  Error saving intermediate image: {e}")
                traceback.print_exc()

            # Save Gemini scores
            if best_group_data["gemini_scores"]:
                try:
                    with open(gemini_txt_path, 'w') as f:
                        json.dump(best_group_data["gemini_scores"], f, indent=4)
                    print(f"  Saved Gemini output to {gemini_txt_path}")
                except Exception as e:
                    print(f"  Error saving Gemini output: {e}")
            else:
                 print(f"  Skipping Gemini output save (no scores).")

        # --- Stage 2: Reconstruction ---
        print(f"  Starting reconstruction...")
        images_for_recon = best_group_data['images_tensor'].to(device) # Use the selected tensor (1, 6, 3, H, W)
        images_for_recon = v2.functional.resize(images_for_recon, 320, interpolation=3, antialias=True).clamp(0, 1)

        # --- Camera Selection ---
        # Create cameras ONCE outside the loop, select here
        base_input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale)
        current_input_cameras = base_input_cameras # Start with all 6
        if args.view == 4:
            print("  Selecting 4 views for reconstruction...")
            indices = torch.tensor([0, 2, 4, 5]).long() # Standard 4 views
            images_for_recon = images_for_recon[:, indices]
            current_input_cameras = base_input_cameras[:, indices] # Select corresponding cameras
        current_input_cameras = current_input_cameras.to(device) # Move selected cameras to device

        # --- Mesh Extraction ---
        with torch.no_grad():
            print("  Generating triplanes...")
            # Ensure model and cameras are on the correct device
            planes = model.to(device).forward_planes(images_for_recon, current_input_cameras)
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
                    save_obj_with_mtl(
                        vertices.data.cpu().numpy(), uvs.data.cpu().numpy(), faces.data.cpu().numpy(),
                        mesh_tex_idx.data.cpu().numpy(), tex_map.permute(1, 2, 0).data.cpu().numpy(),
                        output_obj_path
                    )
                else:
                    print("  Warning: export_texmap requested but mesh output format unexpected. Saving with vertex colors.")
                    vertices, faces, vertex_colors = mesh_out # Fallback assuming vertex colors
                    # --- Safely convert ALL components to NumPy --- 
                    verts_np = np.asarray(vertices) if isinstance(vertices, memoryview) else (vertices.data.cpu().numpy() if hasattr(vertices, 'data') else np.array(vertices))
                    faces_np = np.asarray(faces) if isinstance(faces, memoryview) else (faces.data.cpu().numpy() if hasattr(faces, 'data') else np.array(faces))
                    
                    if isinstance(vertex_colors, memoryview):
                        colors_np = np.asarray(vertex_colors)
                    elif hasattr(vertex_colors, 'data'): # Check for tensor-like
                        colors_np = vertex_colors.data.cpu().numpy()
                    elif isinstance(vertex_colors, np.ndarray):
                        colors_np = vertex_colors # Already numpy
                    else:
                        print(f"  Warning: Unexpected type for vertex_colors: {type(vertex_colors)}. Using fallback gray.")
                        colors_np = np.ones_like(verts_np) * 0.5 
                    # --- End Safe Conversion ---
                    save_obj(verts_np, faces_np, colors_np, output_obj_path)

            else:
                # Ensure mesh_out has vertex colors
                if len(mesh_out) == 3:
                     vertices, faces, vertex_colors = mesh_out
                     # --- Safely convert ALL components to NumPy --- 
                     verts_np = np.asarray(vertices) if isinstance(vertices, memoryview) else (vertices.data.cpu().numpy() if hasattr(vertices, 'data') else np.array(vertices))
                     faces_np = np.asarray(faces) if isinstance(faces, memoryview) else (faces.data.cpu().numpy() if hasattr(faces, 'data') else np.array(faces))
                     
                     if isinstance(vertex_colors, memoryview):
                         colors_np = np.asarray(vertex_colors)
                     elif hasattr(vertex_colors, 'data'): # Check for tensor-like
                         colors_np = vertex_colors.data.cpu().numpy()
                     elif isinstance(vertex_colors, np.ndarray):
                         colors_np = vertex_colors # Already numpy
                     else:
                         print(f"  Warning: Unexpected type for vertex_colors: {type(vertex_colors)}. Using fallback gray.")
                         colors_np = np.ones_like(verts_np) * 0.5 
                     # --- End Safe Conversion ---
                     save_obj(verts_np, faces_np, colors_np, output_obj_path)
                else:
                     print("  Warning: Vertex colors expected but mesh output format unexpected. Saving without colors.")
                     # Handle cases where only vertices and faces might be returned
                     if len(mesh_out) == 2:
                          vertices, faces = mesh_out
                          # --- Safely convert ALL components to NumPy --- 
                          verts_np = np.asarray(vertices) if isinstance(vertices, memoryview) else (vertices.data.cpu().numpy() if hasattr(vertices, 'data') else np.array(vertices))
                          faces_np = np.asarray(faces) if isinstance(faces, memoryview) else (faces.data.cpu().numpy() if hasattr(faces, 'data') else np.array(faces))
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
    # (Keep the existing logic that calls process_image based on args.batch_mode
    # and uses input_file_to_process_single for single mode)
    if args.batch_mode:
        # Input path validity established above using is_effectively_input_dir
        input_dir = args.input_path
        # Use glob directly again here as it's more reliable with Drive mount issues
        all_images = sorted(glob(os.path.join(input_dir, '*.png')))
        if not all_images:
             # Check again in case glob failed but subprocess didn't provide list
             print(f"Error: No PNG images found in batch input directory via glob: {input_dir}. Subprocess check might have passed on empty dir.")
             exit(1)
        print(f"--- Starting Batch Mode: Found {len(all_images)} PNG images in {input_dir} ---")
        
        for img_path in tqdm(all_images, desc="Processing Batch"): 
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            intermediate_subdir = os.path.join(args.output_intermediate_path, f'data_{img_name}')
            output_subdir = os.path.join(args.output_3d_path, f'output_{img_name}')
            os.makedirs(intermediate_subdir, exist_ok=True)
            os.makedirs(output_subdir, exist_ok=True)
            
            # Pass camera creation logic inside or handle it here if needed
            process_image(args, config, model_config, infer_config, device, 
                          pipeline, model, gemini_verifier, rembg_session, None, # Pass None for cameras
                          img_path, intermediate_subdir, output_subdir, is_gemini_pass=False)
            
            if gemini_available_flag:
                 process_image(args, config, model_config, infer_config, device, 
                               pipeline, model, gemini_verifier, rembg_session, None, # Pass None for cameras
                               img_path, intermediate_subdir, output_subdir, is_gemini_pass=True)
            else:
                 print(f"  [{img_name}] Skipping Gemini pass (verifier not available/enabled).")

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
