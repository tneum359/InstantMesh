import os
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# --- Add parent directory to sys.path --- Added
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR) # This is the parent repo root
# ROOT_DIR = os.path.dirname(PARENT_DIR) # Removed incorrect calculation
sys.path.append(PARENT_DIR) # Changed from ROOT_DIR
print(f"--- DEBUG: Added {PARENT_DIR} to sys.path ---") # Updated debug print
# --- End sys.path modification ---

# --- Added imports from scaling.py ---
import json
from datetime import datetime
import random
from dotenv import load_dotenv
from io import BytesIO
import base64
import typing
# Assuming verifier scripts are in a 'verifiers' subdirectory or python path
try:
    from verifiers.gemini_verifier import GeminiVerifier
    from verifiers.laion_aesthetics import LAIONAestheticVerifier
except ImportError:
    print("Warning: Failed to import verifiers. Make sure 'gemini_verifier.py' and 'laion_aesthetics.py' are accessible.")
    GeminiVerifier = None
    LAIONAestheticVerifier = None
# --- End added imports ---

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video


# --- Helper function from scaling.py ---
def convert_to_bytes(image: Image.Image, b64_encode=False) -> typing.Union[bytes, str]:
    """Converts a PIL Image to bytes (PNG format for Gemini)."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    if b64_encode:
        return base64.b64encode(img_bytes).decode("utf-8")
    return img_bytes
# --- End helper function ---


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
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


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames


###############################################################################
# Arguments.
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('input_path', type=str, help='Path to input image or directory.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
# --- Added arguments ---
parser.add_argument('--num_candidates', type=int, default=1, help='Number of candidate multiview groups to generate and evaluate.')
parser.add_argument('--gemini_prompt', type=str, default="A high-quality 3D render of the object.", help='Prompt for Gemini verifier.')
# --- End added arguments ---
args = parser.parse_args()
seed_everything(args.seed)

###############################################################################
# Stage 0: Configuration.
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')

# --- Load Gemini API Key ---
load_dotenv()
use_gemini = args.num_candidates > 1
if use_gemini and not os.getenv("GEMINI_API_KEY"):
    print("Warning: GEMINI_API_KEY not found in .env file. Gemini verification will be skipped.")
    use_gemini = False
if use_gemini and GeminiVerifier is None:
     print("Warning: GeminiVerifier not imported correctly. Gemini verification will be skipped.")
     use_gemini = False
# --- End Load Gemini API Key ---


# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
print('Loading custom white-background unet ...')
if os.path.exists(infer_config.unet_path):
    unet_ckpt_path = infer_config.unet_path
else:
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device)

# --- Initialize Verifiers ---
gemini_verifier = None
laion_verifier = None
if use_gemini:
    print("Initializing Gemini verifier...")
    try:
        gemini_verifier = GeminiVerifier() # Requires GEMINI_API_KEY
    except Exception as e:
        print(f"Error initializing GeminiVerifier: {e}. Disabling Gemini verification.")
        use_gemini = False

# Initialize LAION verifier regardless (can be used for info even if Gemini selects)
print("Initializing LAION Aesthetic verifier...")
try:
    if LAIONAestheticVerifier:
        laion_verifier = LAIONAestheticVerifier(dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    else:
        print("Warning: LAIONAestheticVerifier not imported.")
except Exception as e:
    print(f"Error initializing LAIONAestheticVerifier: {e}")
    laion_verifier = None
# --- End Initialize Verifiers ---


# load reconstruction model
print('Loading reconstruction model ...')
model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'renderer' not in k} # Avoid loading renderer part if present
model.load_state_dict(state_dict, strict=False) # Use strict=False if only loading LRM part

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

# make output directories
output_root = os.path.join(args.output_path, config_name)
image_path = os.path.join(output_root, 'images') # For final selected images + input
intermediate_path = os.path.join(output_root, 'intermediate_views') # For saving best candidate views
mesh_path = os.path.join(output_root, 'meshes')
video_path = os.path.join(output_root, 'videos')
os.makedirs(image_path, exist_ok=True)
os.makedirs(intermediate_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# process input files
if os.path.isdir(args.input_path):
    input_files = [
        os.path.join(args.input_path, file) 
        for file in os.listdir(args.input_path) 
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp')
    ]
else:
    input_files = [args.input_path]
print(f'Total number of input images: {len(input_files)}')


###############################################################################
# Stage 1: Multiview generation and Selection.
###############################################################################

rembg_session = None if args.no_rembg else rembg.new_session()

outputs_for_stage2 = [] # Store data for the selected best views
all_run_results = [] # Store metadata for all inputs

for idx, image_file in enumerate(input_files):
    start_time = datetime.now()
    name = os.path.basename(image_file).split('.')[0]
    print(f'[{idx+1}/{len(input_files)}] Processing {name} ...')

    # remove background optionally
    print("  Preprocessing input image...")
    input_image = Image.open(image_file)
    if not args.no_rembg:
        input_image = remove_background(input_image, rembg_session)
        # Optional: Resize foreground (check if needed for zero123plus)
        # input_image = resize_foreground(input_image, 0.85)

    # --- Candidate Generation Loop ---
    best_group_data = {
        "avg_score": -float('inf'), # Initialize lower for maximization
        "images_pil": None,         # Store PIL images for saving
        "images_tensor": None,      # Store Tensor for reconstruction
        "gemini_scores": None,
        "laion_scores": None,
        "seed": -1
    }

    print(f"  Generating and evaluating {args.num_candidates} candidate groups...")
    for i in range(args.num_candidates):
        print(f"    --- Candidate Group {i+1}/{args.num_candidates} ---")
        current_seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(current_seed)

        # sampling - Generate all 6 views at once
        print(f"    Generating multiview images with seed: {current_seed}...")
        try:
            output_image_pil = pipeline(
                input_image,
                num_inference_steps=args.diffusion_steps,
                generator=generator, # Use seeded generator
            ).images[0]
            # output_image_pil is a 960x640 image with 6 views (3x2 grid)
        except Exception as e:
            print(f"    Error during image generation for group {i+1}: {e}")
            continue # Skip to the next candidate group

        # Convert PIL image grid to tensor of individual images
        images_np = np.asarray(output_image_pil, dtype=np.float32) / 255.0
        images_tensor = torch.from_numpy(images_np).permute(2, 0, 1).contiguous().float() # (3, 960, 640)
        images_tensor = rearrange(images_tensor, 'c (n h) (m w) -> (n m) c h w', n=3, m=2) # (6, 3, 320, 320)

        # --- Verification ---
        avg_group_score = -1.0 # Default score if verification fails or is skipped
        current_gemini_results = []
        current_laion_results = []

        # Convert tensor back to list of PIL Images for verifiers
        # Ensure images are in CPU memory before converting to PIL
        images_pil_list = [v2.functional.to_pil_image(img_t.cpu()) for img_t in images_tensor]

        # Apply Gemini Verifier
        if use_gemini and gemini_verifier:
            print("    Applying Gemini Verifier...")
            gemini_prompts = [args.gemini_prompt] * len(images_pil_list)
            try:
                gemini_inputs = gemini_verifier.prepare_inputs(images=images_pil_list, prompts=gemini_prompts)
                current_gemini_results = gemini_verifier.score(inputs=gemini_inputs) # List of dicts
                group_overall_scores = [res.get("overall_score", {}).get("score", 0) for res in current_gemini_results]
                if group_overall_scores:
                    avg_group_score = np.mean(group_overall_scores)
                else:
                    avg_group_score = 0 # Handle empty results
                print(f"    Gemini Average Overall Score for Group: {avg_group_score:.4f}")
            except Exception as e:
                print(f"    Error during Gemini verification for group {i+1}: {e}")
                avg_group_score = -1 # Penalize groups that fail verification
                current_gemini_results = []

        # Apply LAION Aesthetic Verifier (Optional, but kept for info)
        if laion_verifier:
            print("    Applying LAION Aesthetic Verifier...")
            try:
                # LAION verifier expects tensors on the correct device
                laion_input_tensors = images_tensor.to(device, dtype=laion_verifier.dtype)
                laion_inputs = laion_verifier.prepare_inputs(images=laion_input_tensors) # Pass tensor directly if possible, or handle PIL list conversion inside
                current_laion_results = laion_verifier.score(inputs=laion_inputs)
                avg_laion_score = np.mean([res["laion_aesthetic_score"] for res in current_laion_results])
                print(f"    LAION Average Aesthetic Score for Group: {avg_laion_score:.4f}")
            except Exception as e:
                print(f"    Error during LAION verification for group {i+1}: {e}")
                current_laion_results = []

        # --- Update Best Group ---
        # If not using Gemini, the first candidate is the "best"
        score_to_compare = avg_group_score if use_gemini else i # Use index 'i' to select first candidate if gemini off

        # Update if current score is better OR if it's the first candidate when not using Gemini
        if (use_gemini and score_to_compare > best_group_data["avg_score"]) or (not use_gemini and i == 0):
            print(f"    New best group found with score: {score_to_compare:.4f}")
            best_group_data["avg_score"] = score_to_compare
            best_group_data["images_pil"] = output_image_pil # Save the grid PIL image
            best_group_data["images_tensor"] = images_tensor # Save the tensor (6, C, H, W)
            best_group_data["gemini_scores"] = current_gemini_results
            best_group_data["laion_scores"] = current_laion_results
            best_group_data["seed"] = current_seed
    # --- End Candidate Generation Loop ---

    # --- Process Best Group ---
    if best_group_data["images_tensor"] is None:
        print(f"  No successful candidate groups generated or verified for {name}. Skipping reconstruction.")
        continue # Skip to the next input image

    print(f"  Selected best group for {name} (Seed: {best_group_data['seed']}, Score: {best_group_data['avg_score']:.4f})")

    # Save the best PIL image grid (intermediate output)
    intermediate_filename = os.path.join(intermediate_path, f'{name}_seed{best_group_data["seed"]}_score{best_group_data["avg_score"]:.2f}.png')
    try:
        best_group_data["images_pil"].save(intermediate_filename)
        print(f"  Best intermediate view grid saved to {intermediate_filename}")
    except Exception as e:
        print(f"  Error saving intermediate image: {e}")

    # Prepare data for Stage 2 (Reconstruction)
    outputs_for_stage2.append({
        'name': name,
        'images': best_group_data["images_tensor"] # Pass the tensor (6, C, H, W)
    })

    # Save input image (processed)
    input_image.save(os.path.join(image_path, f'{name}_input.png'))

    # Save results summary for this input
    avg_laion_score_best_group = np.mean([res["laion_aesthetic_score"] for res in best_group_data["laion_scores"]]) if best_group_data["laion_scores"] else None
    gemini_explanation = "N/A"
    if best_group_data["gemini_scores"] and isinstance(best_group_data["gemini_scores"], list) and len(best_group_data["gemini_scores"]) > 0:
        first_img_scores = best_group_data["gemini_scores"][0]
        if isinstance(first_img_scores, dict):
             gemini_explanation = first_img_scores.get('overall_score', {}).get('explanation', "N/A")

    run_result_data = {
        "input_name": name,
        "input_file": image_file,
        "config": config_name,
        "num_candidates": args.num_candidates,
        "best_seed": best_group_data["seed"],
        "best_gemini_avg_score": best_group_data["avg_score"] if use_gemini else None,
        "best_laion_avg_score": avg_laion_score_best_group,
        "gemini_prompt": args.gemini_prompt if use_gemini else None,
        "representative_gemini_explanation": gemini_explanation if use_gemini else None,
        "intermediate_views_file": intermediate_filename,
        "processing_time_seconds": (datetime.now() - start_time).total_seconds()
    }
    all_run_results.append(run_result_data)

# Save overall results JSON
results_json_path = os.path.join(output_root, 'run_summary.json')
try:
    with open(results_json_path, 'w') as f:
        json.dump(all_run_results, f, indent=4)
    print(f"\nRun summary saved to {results_json_path}")
except Exception as e:
    print(f"\nError saving run summary JSON: {e}")


# --- Optional: Delete pipeline and verifiers to save memory before reconstruction ---
print("\nDeleting diffusion pipeline and verifiers to free memory...")
try:
    del pipeline
    del gemini_verifier
    del laion_verifier
    if 'images_pil_list' in locals(): del images_pil_list # Clean up intermediate vars
    if 'laion_input_tensors' in locals(): del laion_input_tensors
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  Error during cleanup: {e}")
# --- End Optional Cleanup ---


###############################################################################
# Stage 2: Reconstruction.
###############################################################################

print(f"\n--- Starting Stage 2: Reconstruction for {len(outputs_for_stage2)} selected inputs ---")

# Get input cameras for the reconstruction model (likely expects 6 views)
# Adjust radius based on args.scale if necessary
input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device) # Uses 4.0 radius by default
chunk_size = 20 if IS_FLEXICUBES else 1

for idx, sample in enumerate(outputs_for_stage2):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs_for_stage2)}] Creating mesh for {name} ...')

    # Images are already (6, C, H, W) tensor from Stage 1
    images = sample['images'].unsqueeze(0).to(device) # Add batch dim -> (1, 6, C, H, W)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    # If --view 4 was specified, select the standard 4 views for reconstruction
    # Note: Stage 1 generated 6 views regardless, we select here for reconstruction
    current_input_cameras = input_cameras
    if args.view == 4:
        print("  Selecting 4 views for reconstruction...")
        indices = torch.tensor([0, 2, 4, 5]).long() # Standard 4 views indices for Zero123++
        images = images[:, indices]
        current_input_cameras = input_cameras[:, indices] # Select corresponding cameras

    with torch.no_grad():
        # get triplane
        print("  Generating triplanes...")
        # Ensure model's forward_planes expects (B, N, C, H, W) and camera format (B, N, CamDim)
        planes = model.forward_planes(images, current_input_cameras)

        # get mesh
        print("  Extracting mesh...")
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

        # Adjust mesh extraction based on model type
        try:
            mesh_out = model.extract_mesh(
                planes,
                use_texture_map=args.export_texmap,
                **infer_config,
            )
        except TypeError as te:
             # Handle potential API changes in extract_mesh, e.g., unexpected kwargs
             print(f"  Warning: TypeError during mesh extraction: {te}. Trying without infer_config...")
             try:
                mesh_out = model.extract_mesh(planes, use_texture_map=args.export_texmap)
             except Exception as e_fallback:
                 print(f"  Error during fallback mesh extraction: {e_fallback}. Skipping mesh saving.")
                 continue # Skip to video rendering or next sample


        print(f"  Saving mesh to {mesh_path_idx}...")
        if args.export_texmap:
            try:
                vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                save_obj_with_mtl(
                    vertices.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map.permute(1, 2, 0).data.cpu().numpy(),
                    mesh_path_idx,
                )
            except Exception as e_save:
                print(f"  Error saving mesh with texture map: {e_save}")
        else:
            try:
                # Assuming output is (vertices, faces, vertex_colors) for non-texmap case
                vertices, faces, vertex_colors = mesh_out
                save_obj(vertices, faces, vertex_colors, mesh_path_idx)
            except Exception as e_save:
                 print(f"  Error saving mesh with vertex colors: {e_save}")
        print(f"  Mesh saved to {mesh_path_idx}")

        # get video
        if args.save_video:
            print("  Rendering video...")
            video_path_idx = os.path.join(video_path, f'{name}.mp4')
            render_size = infer_config.get('render_resolution', 512) # Use get with default
            try:
                render_cameras = get_render_cameras(
                    batch_size=1,
                    M=120,
                    radius=args.distance, # Use distance argument for rendering
                    elevation=20.0,
                    is_flexicubes=IS_FLEXICUBES,
                ).to(device)

                frames = render_frames(
                    model,
                    planes,
                    render_cameras=render_cameras,
                    render_size=render_size,
                    chunk_size=chunk_size,
                    is_flexicubes=IS_FLEXICUBES,
                )

                save_video(
                    frames,
                    video_path_idx,
                    fps=30,
                )
                print(f"  Video saved to {video_path_idx}")
            except Exception as e_render:
                print(f"  Error rendering or saving video: {e_render}")

print("\n--- Script Finished ---")
