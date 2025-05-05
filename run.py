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

# --- Google Drive Mount --- Added
try:
    from google.colab import drive
    print("Attempting to mount Google Drive...")
    try:
        drive.mount('/content/drive')
        # Define Google Drive base paths
        DRIVE_BASE_PATH = '/content/drive/MyDrive/final_project' # CHANGE THIS IF YOUR FOLDER IS DIFFERENT
        DRIVE_INPUT_PATH = os.path.join(DRIVE_BASE_PATH, 'input_images')
        DRIVE_INTERMEDIATE_PATH = os.path.join(DRIVE_BASE_PATH, 'intermediate_images')
        DRIVE_OUTPUT_3D_PATH = os.path.join(DRIVE_BASE_PATH, 'output_3d')
        print(f"Using Google Drive paths:\n  Input: {DRIVE_INPUT_PATH}\n  Intermediate: {DRIVE_INTERMEDIATE_PATH}\n  Output 3D: {DRIVE_OUTPUT_3D_PATH}")
        # Ensure base output dirs exist
        os.makedirs(DRIVE_INTERMEDIATE_PATH, exist_ok=True)
        os.makedirs(DRIVE_OUTPUT_3D_PATH, exist_ok=True)
        IS_COLAB = True
    except Exception as e:
        print(f"Warning: Failed to mount Google Drive: {e}")
        print("Falling back to local paths.")
        IS_COLAB = False
        DRIVE_INPUT_PATH = None
        DRIVE_INTERMEDIATE_PATH = 'outputs/intermediate_images'
        DRIVE_OUTPUT_3D_PATH = 'outputs/output_3d'
        os.makedirs(DRIVE_INTERMEDIATE_PATH, exist_ok=True)
        os.makedirs(DRIVE_OUTPUT_3D_PATH, exist_ok=True)
except ImportError:
    print("Google Colab not detected. Using local paths.")
    IS_COLAB = False
    # Define fallback local paths
    DRIVE_INPUT_PATH = None # Requires input_path argument
    DRIVE_INTERMEDIATE_PATH = 'outputs/intermediate_images'
    DRIVE_OUTPUT_3D_PATH = 'outputs/output_3d'
    os.makedirs(DRIVE_INTERMEDIATE_PATH, exist_ok=True)
    os.makedirs(DRIVE_OUTPUT_3D_PATH, exist_ok=True)

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
    print("Successfully imported verifiers.")
except ImportError as e:
    print(f"Warning: Failed to import verifiers ({e}). Make sure 'verifiers' directory is accessible from {PARENT_DIR}.")
    GeminiVerifier = None
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
parser.add_argument('--input_path', type=str, default=DRIVE_INPUT_PATH, help='Path to input image directory (defaults to GDrive path if mounted).')
parser.add_argument('--output_intermediate_path', type=str, default=DRIVE_INTERMEDIATE_PATH, help='Base directory for intermediate outputs (defaults to GDrive path if mounted).')
parser.add_argument('--output_3d_path', type=str, default=DRIVE_OUTPUT_3D_PATH, help='Base directory for final 3D outputs (defaults to GDrive path if mounted).')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views for reconstruction.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
parser.add_argument('--num_candidates', type=int, default=1, help='Number of candidate multiview groups to generate and evaluate.')
parser.add_argument('--gemini_prompt', type=str, default=None, help='Prompt for Gemini verifier (defaults to reading from verifiers/verifier_prompt.txt).')
args = parser.parse_args()

# --- Load default Gemini prompt from file if not provided --- Added
DEFAULT_GEMINI_PROMPT_FALLBACK = "Evaluate the quality of the generated 3D object views based on realism, detail, consistency, and adherence to the likely subject."
if args.gemini_prompt is None:
    prompt_file_path = os.path.join(PARENT_DIR, "verifiers", "verifier_prompt.txt")
    print(f"--gemini_prompt not provided, attempting to load from {prompt_file_path}")
    try:
        with open(prompt_file_path, 'r') as f:
            args.gemini_prompt = f.read()
        print("  Successfully loaded default prompt from file.")
    except FileNotFoundError:
        print(f"  Warning: Default prompt file not found at {prompt_file_path}. Using fallback default.")
        args.gemini_prompt = DEFAULT_GEMINI_PROMPT_FALLBACK
    except Exception as e:
        print(f"  Warning: Error reading default prompt file {prompt_file_path}: {e}. Using fallback default.")
        args.gemini_prompt = DEFAULT_GEMINI_PROMPT_FALLBACK
else:
    print(f"Using provided --gemini_prompt.")
# --- End loading default prompt ---

# Ensure input path is provided if not using default GDrive path
if args.input_path is None and not IS_COLAB:
    parser.error("--input_path is required when not running in Google Colab or GDrive not mounted.")
# Ensure output paths are valid
if not os.path.isdir(args.output_intermediate_path):
    os.makedirs(args.output_intermediate_path, exist_ok=True)
if not os.path.isdir(args.output_3d_path):
    os.makedirs(args.output_3d_path, exist_ok=True)

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
    print("Warning: GeminiVerifier not available. Gemini verification will be skipped.")
    use_gemini = False

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

# Initialize verifiers if needed
gemini_verifier = None
if use_gemini:
    print("Initializing Gemini Verifier...")
    try:
        gemini_verifier = GeminiVerifier()
        print("Gemini Verifier initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini Verifier: {e}")
        use_gemini = False

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

# process input files
input_image_dir = args.input_path
if not os.path.isdir(input_image_dir):
    print(f"Error: Input path '{input_image_dir}' is not a valid directory.")
    sys.exit(1)

input_files = sorted([ # Sort for consistent naming (e.g., 01, 02)
    os.path.join(input_image_dir, file)
    for file in os.listdir(input_image_dir)
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
])

if not input_files:
    print(f"Error: No valid image files found in '{input_image_dir}'.")
    sys.exit(1)

print(f'Total number of input images found: {len(input_files)}')


###############################################################################
# Stage 1: Multiview generation and Selection.
###############################################################################

rembg_session = None if args.no_rembg else rembg.new_session()

outputs_for_stage2 = [] # Store data for the selected best views

for idx, image_file in enumerate(input_files):
    start_time = datetime.now()
    # Extract base name without extension (e.g., "image_01")
    name = os.path.splitext(os.path.basename(image_file))[0]
    print(f'\n[{idx+1}/{len(input_files)}] Processing {name} (from {image_file}) ...')

    # --- Define output paths for this specific input --- Added
    intermediate_subdir = os.path.join(args.output_intermediate_path, name)
    os.makedirs(intermediate_subdir, exist_ok=True)

    intermediate_image_path = os.path.join(intermediate_subdir, f'intermediate_{name}.png')
    gemini_txt_path = os.path.join(intermediate_subdir, f'gemini_output_{name}.txt')
    output_obj_path = os.path.join(args.output_3d_path, f'generation_{name}.obj')
    output_video_path = os.path.join(intermediate_subdir, f'video_{name}.mp4') # Video saved in intermediate dir

    print(f"  Intermediate outputs will be saved to: {intermediate_subdir}")
    print(f"  Final 3D object will be saved to: {output_obj_path}")
    # --- End path definition ---

    # remove background optionally
    print("  Preprocessing input image...")
    try:
        input_image = Image.open(image_file)
    except Exception as e:
        print(f"  Error opening image {image_file}: {e}. Skipping.")
        continue

    if not args.no_rembg:
        print("  Removing background...")
        try:
            input_image = remove_background(input_image, rembg_session)
        except Exception as e:
            print(f"  Error removing background: {e}. Proceeding with original image.")
        # Optional: Resize foreground (check if needed for zero123plus)
        # input_image = resize_foreground(input_image, 0.85)


    # --- Candidate Generation Loop ---
    best_group_data = {
        "avg_score": -float('inf'),
        "images_pil": None,        # Store PIL grid of BEST candidate
        "images_tensor": None,     # Store Tensor of BEST candidate for reconstruction
        "gemini_scores": None,     # Store Gemini JSON output for BEST candidate
        "seed": -1
    }

    print(f"  Generating and evaluating {args.num_candidates} candidate groups...")
    for i in range(args.num_candidates):
        print(f"    --- Candidate Group {i+1}/{args.num_candidates} ---")
        current_seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=device).manual_seed(current_seed)
        output_image_pil = None

        print(f"    Generating multiview images with seed: {current_seed}...")
        try:
            output_image_pil = pipeline(
                input_image,
                num_inference_steps=args.diffusion_steps,
                generator=generator,
            ).images[0]
        except Exception as e:
            print(f"    Error during image generation for group {i+1}: {e}")
            continue

        if output_image_pil is None:
            print(f"    Skipping candidate {i+1} due to generation failure.")
            continue

        # Split the grid image into 6 separate views (assuming 3 rows x 2 columns)
        w, h = output_image_pil.size
        single_w, single_h = w // 2, h // 3
        images_pil_list = [
            output_image_pil.crop((col * single_w, row * single_h, (col + 1) * single_w, (row + 1) * single_h))
            for row in range(3) for col in range(2)
        ]

        # Remove background from each generated view using the same function as input preprocessing
        images_pil_list = [remove_background(img, rembg_session=rembg_session) for img in images_pil_list]

        # Convert cleaned PIL images back to tensor for reconstruction and saving
        to_tensor = v2.ToTensor()
        images_tensor = torch.stack([to_tensor(img.convert('RGB')) for img in images_pil_list])
        images_tensor = images_tensor.unsqueeze(0)
        print("images_tensor shape:", images_tensor.shape, "dtype:", images_tensor.dtype, "min:", images_tensor.min().item(), "max:", images_tensor.max().item())

        if use_gemini and gemini_verifier:
            print("    Applying Gemini Verifier to evaluate multiview set...")
            try:
                # Prepare inputs for all 6 views at once
                gemini_inputs = gemini_verifier.prepare_inputs(
                    images=images_pil_list,
                    prompts=[args.gemini_prompt] * 6  # Same prompt for all views
                )
                
                # Get evaluation for the entire set
                gemini_result = gemini_verifier.score(inputs=gemini_inputs)
                
                if gemini_result["success"]:
                    result = gemini_result["result"]
                    avg_group_score = result["overall_score"]
                    print(f"    Gemini Evaluation Results:")
                    print(f"      Aesthetic Quality: {result['aesthetic_quality']['score']:.2f}")
                    print(f"      Visual Consistency: {result['visual_consistency']['score']:.2f}")
                    print(f"      Reconstruction Potential: {result['reconstruction_potential']['score']:.2f}")
                    print(f"      Overall Score: {avg_group_score:.2f}")
                    print(f"      Assessment: {result['overall_assessment']}")
                else:
                    error_msg = gemini_result.get('error', 'Unknown error') if isinstance(gemini_result, dict) else str(gemini_result)
                    print(f"    Error in Gemini evaluation: {error_msg}")
                    avg_group_score = -1
                    gemini_result = None
            except Exception as e:
                print(f"    Error during Gemini verification for group {i+1}: {e}")
                avg_group_score = -1
                gemini_result = None

        # --- Update Best Group ---
        score_to_compare = avg_group_score if use_gemini else i
        if (use_gemini and score_to_compare > best_group_data["avg_score"]) or (not use_gemini and i == 0):
            print(f"    New best group found with score: {score_to_compare:.4f}")
            best_group_data["avg_score"] = score_to_compare
            best_group_data["images_pil"] = images_pil_list # Save best PIL grid
            best_group_data["images_tensor"] = images_tensor
            best_group_data["gemini_scores"] = gemini_result # Save best scores JSON
            best_group_data["seed"] = current_seed
    # --- End Candidate Generation Loop --


    # --- Process and Save Best Group's Outputs --- Modified
    if best_group_data["images_tensor"] is None:
        print(f"  No successful candidate groups generated or verified for {name}. Skipping reconstruction.")
        # Optional: Log skipped files
        continue

    print(f"  Selected best group for {name} (Seed: {best_group_data['seed']}, Score: {best_group_data['avg_score']:.4f})")

    # Save the best intermediate image grid
    try:
        best_group_data["images_pil"].save(intermediate_image_path)
        print(f"  Saved best intermediate view grid to {intermediate_image_path}")
    except Exception as e:
        print(f"  Error saving best intermediate image: {e}")

    # Save the best Gemini scores
    if use_gemini and best_group_data["gemini_scores"]:
        try:
            with open(gemini_txt_path, 'w') as f:
                json.dump(best_group_data["gemini_scores"], f, indent=4)
            print(f"  Saved Gemini output to {gemini_txt_path}")
        except Exception as e:
            print(f"  Error saving Gemini output: {e}")
    elif use_gemini:
        print(f"  Skipping Gemini output save (no scores available or Gemini disabled).")


    # Prepare data for Stage 2 (Reconstruction) - Pass the determined OBJ path
    outputs_for_stage2.append({
        'name': name,
        'images': best_group_data["images_tensor"],
        'output_obj_path': output_obj_path, # Pass the target save path
        'output_video_path': output_video_path # Pass video path too
    })


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
    mesh_path_idx = sample['output_obj_path'] # Use the path determined in Stage 1
    video_path_idx = sample['output_video_path'] # Use the path determined in Stage 1
    print(f'[{idx+1}/{len(outputs_for_stage2)}] Creating mesh for {name} -> {mesh_path_idx}')

    # Images are already (1, 6, 3, H, W) tensor from Stage 1
    images = sample['images']
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
            render_size = infer_config.get('render_resolution', 512)
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
