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
import shutil # Added for rmtree

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

# --- Core Processing Function ---
def process_image(args, config, model_config, infer_config, device, 
                  pipeline, model, gemini_verifier, rembg_session, input_cameras,
                  input_image_path, intermediate_dir, output_dir, is_gemini_pass):
    """
    Processes a single input image: loads, (optionally) removes background, 
    generates multiview images using Zero123++, (optionally) scores with Gemini, 
    selects the best candidate, and reconstructs a 3D mesh.
    Saves intermediate images and the final 3D object.
    """
    base_img_name = os.path.basename(input_image_path)
    img_name_base = os.path.splitext(base_img_name)[0]

    # --- Output Path Setup ---
    # Ensure per-image subdirectories exist for intermediate and final outputs
    # These are already created in the main block before calling this function.
    # Intermediate dir for this image (e.g., intermediate_images/data_0002)
    # Output dir for this image (e.g., output_3d/output_0002)

    final_obj_name = 'obj_with_gemini.obj' if is_gemini_pass else 'obj_no_gemini.obj'
    final_output_obj_path = os.path.join(output_dir, final_obj_name) # Final name for the .obj file
    output_obj_path = os.path.join(output_dir, f'_{final_obj_name}_temp') # Temp name during generation

    # --- Clear Intermediate Directory for Gemini Pass ---
    if is_gemini_pass and os.path.exists(intermediate_dir):
        print(f"  Clearing previous contents of intermediate directory: {intermediate_dir}")
        for item in os.listdir(intermediate_dir):
            item_path = os.path.join(intermediate_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                        print(f'  Warning: Failed to delete {item_path}. Reason: {e}')
    elif not os.path.exists(intermediate_dir):
         os.makedirs(intermediate_dir, exist_ok=True) # Should already exist but defensive

    # --- Load and Preprocess Input Image ---
    print(f"  Loading input image: {input_image_path}")
    try:
        input_image_pil = Image.open(input_image_path)
        if not args.no_rembg and rembg_session:
            print("  Removing background...")
            input_image_pil = rembg.remove(input_image_pil, session=rembg_session)
        input_image_pil = input_image_pil.convert("RGB") # Ensure RGB format
    except Exception as e:
        print(f"  Error loading or preprocessing input image {input_image_path}: {e}")
        return

    input_image_for_pipeline = F.to_tensor(input_image_pil).unsqueeze(0).to(device, dtype=pipeline.dtype if hasattr(pipeline, 'dtype') else torch.float16) # Prepare for pipeline

    # --- Candidate Generation Loop ---
    best_group_score = -1.0  # Initialize with a low score
    best_group_images_tensor = None
    best_group_pil_list_processed_rgb = None
    best_group_pil_grid_raw = None
    best_group_seed = -1
    best_candidate_metadata = None # For storing Gemini scores and other details

    if is_gemini_pass:
        print(f"  Gemini Pass Activated: Min candidates: {args.min_candidates}, Max candidates: {args.max_candidates}, Target score: {args.target_score:.2f}")
        iterations_for_loop = args.max_candidates
    else:
        print("  Non-Gemini Pass: Generating 1 candidate group.")
        iterations_for_loop = 1

    for candidate_idx in range(iterations_for_loop):
        current_run_seed = random.randint(0, 2**32 - 1)
        seed_everything(current_run_seed)
        
        pass_type_info = "Gemini" if is_gemini_pass else "Non-Gemini"
        candidate_num_info = f"{candidate_idx + 1}/{args.max_candidates}" if is_gemini_pass else "1/1"
        print(f"\n  --- ({pass_type_info}) Generating Candidate Group {candidate_num_info} (Seed: {current_run_seed}) ---")

        # --- Diffusion Model (Zero123++) --- 
        try:
            print("    Generating multiview images with Zero123++...")
            with torch.no_grad():
                output_image_pil_grid = pipeline(
                    input_image_for_pipeline,
                    num_inference_steps=args.diffusion_steps,
                    width=args.gen_width,
                    height=args.gen_height,
                    guidance_scale=3.0, # Default from common Zero123++ usage
                ).images[0]
            print("    Multiview generation complete.")
        except Exception as e:
            print(f"    Error during Zero123++ pipeline for candidate {candidate_idx + 1}: {e}")
            traceback.print_exc()
            if is_gemini_pass and (candidate_idx + 1) < args.min_candidates:
                print("    Critical error before min_candidates met. Aborting for this image.")
                return # Abort for this image if critical failure before min_candidates
            print("    Skipping to next candidate due to error.")
            continue # Skip this candidate

        # --- Process and Prepare Candidate Images ---
        try:
            # (Code from original script to split grid, optionally rembg, convert to tensor)
            # Split the 6-image grid into a list of PIL images (raw from diffusion)
            w, h = output_image_pil_grid.size
            img_width, img_height = w // 2, h // (6 // 2)
            multiview_pil_list_raw = []
            for r_idx in range(6 // 2):
                for c_idx in range(2):
                    box = (c_idx * img_width, r_idx * img_height, (c_idx + 1) * img_width, (r_idx + 1) * img_height)
                    multiview_pil_list_raw.append(output_image_pil_grid.crop(box))
            
            # Optional background removal for each view
            multiview_pil_list_nobg_rgba_current_candidate = [] # For RGBA results if needed
            multiview_pil_list_nobg_rgb_current_candidate = []  # For RGB results for Gemini & Recon

            if not args.no_rembg and rembg_session:
                print("    Removing background from generated views...")
                for i, view_pil in enumerate(multiview_pil_list_raw):
                    try:
                        nobg_view_pil = rembg.remove(view_pil, session=rembg_session).convert("RGBA")
                        multiview_pil_list_nobg_rgba_current_candidate.append(nobg_view_pil)
                        # Create RGB version with white background for Gemini/Reconstruction
                        rgb_version = Image.new("RGB", nobg_view_pil.size, (255, 255, 255))
                        rgb_version.paste(nobg_view_pil, mask=nobg_view_pil.split()[3]) # Paste using alpha channel as mask
                        multiview_pil_list_nobg_rgb_current_candidate.append(rgb_version)
                    except Exception as e_rembg:
                        print(f"      Warning: Failed to remove background for view {i}: {e_rembg}. Using raw view.")
                        multiview_pil_list_nobg_rgba_current_candidate.append(view_pil.convert("RGBA")) # Fallback
                        multiview_pil_list_nobg_rgb_current_candidate.append(view_pil.convert("RGB")) # Fallback
            else:
                print("    Skipping background removal for generated views.")
                multiview_pil_list_nobg_rgba_current_candidate = [img.convert("RGBA") for img in multiview_pil_list_raw]
                multiview_pil_list_nobg_rgb_current_candidate = [img.convert("RGB") for img in multiview_pil_list_raw]

            # Convert to tensor for reconstruction model (expecting B, V, C, H, W)
            # Using the RGB list (multiview_pil_list_nobg_rgb_current_candidate)
            images_tensor = torch.stack([
                F.to_tensor(rgba_to_rgb_white(img.convert("RGBA"))) for img in multiview_pil_list_nobg_rgb_current_candidate
            ]).unsqueeze(0).to(device) # (1, 6, 3, H, W)
            print(f"    Processed views to tensor, shape: {images_tensor.shape}")

        except Exception as e:
            print(f"    Error processing views for candidate {candidate_idx + 1}: {e}")
            traceback.print_exc()
            if is_gemini_pass and (candidate_idx + 1) < args.min_candidates:
                print("    Critical error before min_candidates met. Aborting for this image.")
                return
            print("    Skipping to next candidate due to error.")
            continue

        # --- Gemini Scoring (if applicable) ---
        current_score = 0.0 # Default for non-Gemini or if scoring fails
        current_candidate_metadata = None

        if is_gemini_pass and gemini_verifier:
            print(f"    Scoring candidate {candidate_idx + 1} with Gemini...")
            try:
                # Convert RGB PIL images to base64 for Gemini Verifier
                base64_images = [gemini_verifier.pil_to_base64(img) for img in multiview_pil_list_nobg_rgb_current_candidate]
                score, explanation, full_response = gemini_verifier.score_candidate_images(base64_images)
                current_score = score
                current_candidate_metadata = {
                    "gemini_score": score,
                    "gemini_explanation": explanation,
                    "seed": current_run_seed,
                    "candidate_index": candidate_idx + 1
                    # "full_gemini_response": full_response # Optional: can be large
                }
                print(f"    Gemini Score for Candidate {candidate_idx + 1}: {current_score:.2f}")
                if explanation: print(f"    Gemini Rationale: {explanation[:200]}...") # Print a snippet
            except Exception as e_gemini:
                print(f"    Error during Gemini scoring for candidate {candidate_idx + 1}: {e_gemini}")
                current_score = 0.0 # Penalize if Gemini fails for this candidate
                current_candidate_metadata = {"error": str(e_gemini), "seed": current_run_seed, "candidate_index": candidate_idx + 1}
        
        # --- Update Best Candidate --- 
        if current_score > best_group_score or (not is_gemini_pass and candidate_idx == 0): # If non-Gemini, first is always best
            best_group_score = current_score
            best_group_images_tensor = images_tensor.clone() # CRITICAL: Clone the tensor
            best_group_pil_list_processed_rgb = [img.copy() for img in multiview_pil_list_nobg_rgb_current_candidate]
            best_group_pil_grid_raw = output_image_pil_grid.copy()
            best_group_seed = current_run_seed
            if is_gemini_pass and current_candidate_metadata:
                best_candidate_metadata = current_candidate_metadata
            elif not is_gemini_pass: # Store basic seed info for non-Gemini
                 best_candidate_metadata = {"seed": current_run_seed, "candidate_index": candidate_idx + 1, "score": "N/A (Non-Gemini)"}

            print(f"    New best candidate group selected (Score: {best_group_score:.2f}, Seed: {best_group_seed})")

        # --- Early Exit Logic for Gemini Pass ---
        if is_gemini_pass:
            num_candidates_generated = candidate_idx + 1
            if best_group_score >= args.target_score and num_candidates_generated >= args.min_candidates:
                print(f"    Target score {args.target_score:.2f} met or exceeded (Best score: {best_group_score:.2f}) after {num_candidates_generated} candidates (min required: {args.min_candidates}).")
                print("    Stopping candidate generation early.")
                break # Exit the candidate generation loop
            elif num_candidates_generated >= args.max_candidates:
                print(f"    Max candidates ({args.max_candidates}) reached. Best score found: {best_group_score:.2f}.")
                # Loop will terminate naturally
    # --- End Candidate Generation Loop ---

    if best_group_images_tensor is None:
        print(f"  ERROR: No valid candidate group found for {img_name_base} after trying up to {args.max_candidates if is_gemini_pass else 1} candidates. Cannot proceed.")
        # Save any available metadata if a general error occurred before selection
        if is_gemini_pass and not os.path.exists(os.path.join(intermediate_dir, 'general_processing_error.json')):
            try:
                error_meta = {"error_message": "No best candidate selected during processing.", "image_name": img_name_base, "is_gemini_pass": is_gemini_pass}
                with open(os.path.join(intermediate_dir, 'general_processing_error.json'), 'w') as f_err:
                    json.dump(error_meta, f_err, indent=4)
            except Exception as e_meta:
                print(f"  Warning: Failed to save error metadata: {e_meta}")
        return

    print(f"\n  Finished candidate selection for {img_name_base}. Best Seed: {best_group_seed}, Score: {best_group_score:.2f if is_gemini_pass else 'N/A'}")

    # --- Save Intermediate Outputs ONLY FOR THE BEST Group ---
    if best_group_pil_list_processed_rgb:
        # Save individual processed views of the best candidate
        for i, img in enumerate(best_group_pil_list_processed_rgb):
            img.save(os.path.join(intermediate_dir, f'best_view_processed_{i}.png'))
        print(f"  Saved best candidate processed views to {intermediate_dir}")

        # Create and Save Processed Grid of the best candidate
        try:
            transform_to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
            tensors_for_grid = [transform_to_tensor(img) for img in best_group_pil_list_processed_rgb]
            if tensors_for_grid:
                processed_grid_tensor = make_grid(tensors_for_grid, nrow=2)
                processed_grid_pil = F.to_pil_image(processed_grid_tensor)
                processed_grid_path = os.path.join(intermediate_dir, f'best_multiview_grid_processed_seed_{best_group_seed}.png')
                processed_grid_pil.save(processed_grid_path)
                print(f"  Saved best candidate processed grid to {processed_grid_path}")
        except Exception as e:
                print(f"  Warning: Failed to create or save processed grid for best candidate: {e}")
    
    # Save the RAW grid from the diffusion model for the best candidate (optional)
    if best_group_pil_grid_raw:
        raw_grid_path = os.path.join(intermediate_dir, f'best_multiview_grid_raw_seed_{best_group_seed}.png')
        best_group_pil_grid_raw.save(raw_grid_path)
        print(f"  Saved best candidate RAW grid to {raw_grid_path}")

    # Save Gemini scores for the best candidate if it was a Gemini pass and metadata exists
    if is_gemini_pass and best_candidate_metadata:
        best_meta_path = os.path.join(intermediate_dir, f'gemini_scores_best_seed_{best_group_seed}.json')
        try:
            with open(best_meta_path, 'w') as f:
                json.dump(best_candidate_metadata, f, indent=4)
            print(f"  Saved best candidate's Gemini scores to {best_meta_path}")
        except Exception as e:
            print(f"  Warning: Failed to save best candidate's Gemini metadata: {e}")
    # --- End Intermediate Saving for Best --- 

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
