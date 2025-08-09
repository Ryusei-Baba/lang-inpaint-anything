#!/usr/bin/env python3
"""
Text-based Object Removal System - Batch Processing Version

Uses GroundingDINO + SAM + LaMa to perform text-prompt-based object removal
on all images in the input directory.

Usage:
    python lang_inpaint_anything.py --text_prompt "dog"
"""

# Environment variable setup (set before library imports)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Prevent CUDA sync errors
os.environ['MPLBACKEND'] = 'Agg'  # Set matplotlib backend to headless

import argparse
import sys
import cv2
import torch
import numpy as np
import glob
import warnings

# Fix PyTorch 2.6 weights_only=True default issue
# Monkey patch torch.load to use weights_only=False as default
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
import matplotlib
matplotlib.use('Agg')  # Set GUI-free backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Optional, Tuple
import logging

# Path setup
sys.path.insert(0, str(Path(__file__).parent / "Inpaint-Anything"))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

# Import required modules
try:
    from utils import load_img_to_array, save_array_to_img
    from lama_inpaint import inpaint_img_with_lama
    from text_grounding_detector import TextGroundingDetector
    from sam_processor import create_sam_processor
    from config_loader import ConfigLoader
    IMPORTS_OK = True
except ImportError as e:
    print(f"Required module import error: {e}")
    print("Please check if the Inpaint-Anything submodule is properly configured.")
    IMPORTS_OK = False


def setup_output_directories(base_dir: str, subdirs: dict) -> dict:
    """
    Create output directories
    
    Args:
        base_dir: Base directory path
        subdirs: Subdirectory configuration
        
    Returns:
        Dictionary of created directory paths
    """
    output_dirs = {}
    base_path = Path(base_dir)
    
    for key, subdir_name in subdirs.items():
        dir_path = base_path / subdir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        output_dirs[key] = str(dir_path)
        
    return output_dirs


def get_input_images(input_dir: str, supported_formats: List[str]) -> List[str]:
    """
    Get supported image files from input directory
    
    Args:
        input_dir: Input directory path
        supported_formats: List of supported image formats
        
    Returns:
        List of image file paths
    """
    image_files = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Input directory does not exist: {input_dir}")
        return []
    
    for format_ext in supported_formats:
        # Search for lowercase extensions
        pattern = f"*{format_ext}"
        files = glob.glob(str(input_path / pattern))
        image_files.extend(files)
        
        # Search for uppercase extensions
        pattern = f"*{format_ext.upper()}"
        files = glob.glob(str(input_path / pattern))
        image_files.extend(files)
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    print(f"Detected image files: {len(image_files)} files")
    
    return image_files


def save_grounding_dino_result(image: np.ndarray, boxes: torch.Tensor, 
                             phrases: List[str], confidences: torch.Tensor,
                             output_path: str):
    """
    Save GroundingDINO detection results
    
    Args:
        image: Input image
        boxes: Bounding boxes
        phrases: Detected phrases
        confidences: Confidence scores
        output_path: Output path
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    for box, phrase, confidence in zip(boxes, phrases, confidences):
        x1, y1, x2, y2 = box.cpu().numpy()
        width = x2 - x1
        height = y2 - y1
        
        # Draw bounding box
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        
        # Draw label
        plt.text(x1, y1 - 10, f"{phrase}: {confidence:.2f}", 
                fontsize=10, color='red', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.title("GroundingDINO Detection Results")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')  # Ensure all figures are closed


def process_image_with_memory_optimization(image: np.ndarray, mask: np.ndarray, 
                                         lama_config: dict, device: str,
                                         max_size: int = 2048, scale_factor: float = 0.5) -> np.ndarray:
    """
    Process image with memory optimization for large images
    
    Args:
        image: Input image
        mask: Mask for inpainting
        lama_config: LaMa configuration
        device: Device to use
        max_size: Maximum image size before resizing
        scale_factor: Scale factor for resizing
        
    Returns:
        Inpainted image
    """
    original_height, original_width = image.shape[:2]
    
    # Check if image needs resizing
    if original_height > max_size or original_width > max_size:
        new_height = int(original_height * scale_factor)
        new_width = int(original_width * scale_factor)
        
        resized_image = cv2.resize(image, (new_width, new_height))
        resized_mask = cv2.resize(mask.astype(np.uint8), (new_width, new_height))
        
        inpainted_img = inpaint_img_with_lama(
            resized_image, resized_mask,
            lama_config['config_path'], lama_config['checkpoint_path'],
            device=device
        )
        
        # Resize back to original size
        inpainted_img = cv2.resize(inpainted_img, (original_width, original_height))
    else:
        inpainted_img = inpaint_img_with_lama(
            image, mask,
            lama_config['config_path'], lama_config['checkpoint_path'],
            device=device
        )
    
    return inpainted_img


def process_single_image(image_path: str, text_prompt: str, 
                        config: dict, output_dirs: dict,
                        detector: TextGroundingDetector,
                        sam_processor) -> bool:
    """
    Process a single image (for batch processing)
    
    Args:
        image_path: Input image path
        text_prompt: Text prompt
        config: Configuration dictionary
        output_dirs: Output directory dictionary
        detector: GroundingDINO detector
        sam_processor: SAM processor
        
    Returns:
        True if processing succeeded, False if failed
    """
    try:
        input_filename = Path(image_path).stem
        file_extension = Path(image_path).suffix
        
        print(f"\nProcessing: {input_filename}{file_extension}")
        
        # Load image
        image = load_img_to_array(image_path)
        
        # Object detection with GroundingDINO
        boxes, logits, phrases = detector.predict(
            image=image,
            text_prompt=text_prompt,
            box_threshold=config['processing']['box_threshold'],
            text_threshold=config['processing']['text_threshold']
        )
        
        if len(boxes) == 0:
            print(f"  No objects detected: {input_filename}")
            return False
        
        print(f"  Detected objects: {len(boxes)}")
        
        # Save GroundingDINO results
        if config['processing']['save_intermediate']:
            grounding_output_path = os.path.join(
                output_dirs['grounding_dino'], 
                f"{input_filename}{file_extension}"
            )
            save_grounding_dino_result(image, boxes, phrases, logits, grounding_output_path)
        
        # Segmentation with SAM
        masks = sam_processor.predict_masks(image, boxes)
        
        if not masks:
            print(f"  Mask generation failed: {input_filename}")
            return False
        
        # Dilate masks
        if config['processing']['dilate_kernel_size'] > 0:
            masks = sam_processor.dilate_masks(masks, config['processing']['dilate_kernel_size'])
        
        # Combine masks if needed
        if config['processing']['combine_masks'] and len(masks) > 1:
            final_mask = sam_processor.combine_masks(masks)
            masks = [final_mask]
        
        # Save SAM masks
        if config['processing']['save_intermediate']:
            for i, mask in enumerate(masks):
                mask_output_path = os.path.join(
                    output_dirs['sam'], 
                    f"{input_filename}_mask{i}{file_extension}"
                )
                mask_normalized = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
                cv2.imwrite(mask_output_path, mask_normalized)
        
        # Inpainting with LaMa
        lama_config = config['models']['lama']
        best_result = None
        
        for i, mask in enumerate(masks):
            try:
                # Memory optimization
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Process with memory optimization
                inpainted_img = process_image_with_memory_optimization(
                    image, mask, lama_config, config['device']['type'],
                    config['processing'].get('max_image_size', 2048),
                    config['processing'].get('memory_scale_factor', 0.5)
                )
                
                # Save LaMa results
                lama_output_path = os.path.join(
                    output_dirs['lama'], 
                    f"{input_filename}{file_extension}"
                )
                save_array_to_img(inpainted_img, lama_output_path)
                best_result = lama_output_path
                break  # Use the first successful mask result
                
            except torch.cuda.OutOfMemoryError:
                print(f"  GPU memory insufficient, retrying with CPU: {input_filename}")
                torch.cuda.empty_cache()
                
                try:
                    inpainted_img = inpaint_img_with_lama(
                        image, mask,
                        lama_config['config_path'], lama_config['checkpoint_path'],
                        device="cpu"
                    )
                    
                    lama_output_path = os.path.join(
                        output_dirs['lama'], 
                        f"{input_filename}{file_extension}"
                    )
                    save_array_to_img(inpainted_img, lama_output_path)
                    best_result = lama_output_path
                    break
                    
                except Exception as e:
                    print(f"  CPU processing also failed: {e}")
                    continue
                    
            except Exception as e:
                print(f"  Inpainting error: {e}")
                continue
        
        if best_result:
            print(f"  Processing completed: {input_filename}")
            return True
        else:
            print(f"  Inpainting failed: {input_filename}")
            return False
            
    except Exception as e:
        print(f"  Processing error ({input_filename}): {e}")
        return False


def initialize_models(config: dict) -> Tuple[TextGroundingDetector, object]:
    """
    Initialize GroundingDINO and SAM models
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (detector, sam_processor)
    """
    print("\nInitializing models...")
    
    # GroundingDINO
    grounding_config = config['models']['grounding_dino']
    detector = TextGroundingDetector(
        config_path=grounding_config['config_path'],
        checkpoint_path=grounding_config['checkpoint_path'],
        device=config['device']['type']
    )
    
    # SAM
    sam_config = config['models']['sam']
    sam_processor = create_sam_processor(
        model_type=sam_config['model_type'],
        checkpoint_path=sam_config['checkpoint_path'],
        device=config['device']['type']
    )
    
    print("Model initialization completed")
    return detector, sam_processor


def setup_logging(config: dict):
    """
    Setup logging configuration
    
    Args:
        config: Configuration dictionary
    """
    log_level = config.get('logging', {}).get('level', 'WARNING')
    logging.basicConfig(
        level=getattr(logging, log_level), 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Adjust log levels for specific libraries
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    logging.getLogger('saicinpainting').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)


def print_processing_summary(success_count: int, failure_count: int, 
                           total_files: int, output_dirs: dict):
    """
    Print processing summary
    
    Args:
        success_count: Number of successful processings
        failure_count: Number of failed processings
        total_files: Total number of files processed
        output_dirs: Output directories
    """
    print(f"\n=== Processing Complete ===")
    print(f"Successful: {success_count} files")
    print(f"Failed: {failure_count} files")
    print(f"Total processed: {total_files} files")
    
    print(f"\nOutput results:")
    for key, path in output_dirs.items():
        print(f"  {key.upper()}: {path}")
    
    if success_count > 0:
        print(f"\n✅ Batch processing completed successfully!")
    else:
        print(f"\n❌ All processing failed.")


def main():
    """Main function - Batch processing support"""
    if not IMPORTS_OK:
        print("Failed to import required modules.")
        sys.exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Text-based Object Removal System - Batch Processing")
    parser.add_argument("--text_prompt", type=str, 
                       help="Text description of objects to remove (uses config default if omitted)")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--input_dir", type=str, 
                       help="Input directory (overrides config file value)")
    parser.add_argument("--output_dir", type=str, 
                       help="Output directory (overrides config file value)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], 
                       help="Device to use (overrides config file value)")
    
    args = parser.parse_args()
    
    # Load configuration
    print("=== Text-based Object Removal System ===")
    print("Loading configuration file...")
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    # Override configuration with command line arguments
    text_prompt = args.text_prompt if args.text_prompt else config.get('general', {}).get('default_text_prompt')
    
    if not text_prompt:
        print("Error: No text prompt specified. Please use --text_prompt argument or set default_text_prompt in config.yaml.")
        sys.exit(1)
    
    if args.input_dir:
        config['processing']['input_dir'] = args.input_dir
    if args.output_dir:
        config['processing']['output_dir'] = args.output_dir
    if args.device:
        config['device']['type'] = args.device
    
    # Auto-select device
    if config['device']['type'] == "auto":
        config['device']['type'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Display configuration
    print(f"Input directory: {config['processing']['input_dir']}")
    print(f"Text prompt: {text_prompt}")
    print(f"Device: {config['device']['type']}")
    print(f"Output directory: {config['processing']['output_dir']}")
    
    # Setup logging
    setup_logging(config)
    
    # Create output directories
    output_subdirs = config.get('general', {}).get('output_subdirs', {
        'grounding_dino': 'GroundingDINO',
        'sam': 'SAM', 
        'lama': 'LAMA'
    })
    output_dirs = setup_output_directories(config['processing']['output_dir'], output_subdirs)
    
    # Get input image files
    supported_formats = config.get('general', {}).get('supported_image_formats', 
                                                    ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'])
    image_files = get_input_images(config['processing']['input_dir'], supported_formats)
    
    if not image_files:
        print("No processable image files found.")
        sys.exit(1)
    
    # Initialize models
    detector, sam_processor = initialize_models(config)
    
    # Execute batch processing
    print(f"\n=== Batch Processing Started ===")
    print(f"Files to process: {len(image_files)}")
    
    success_count = 0
    failure_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end=" ")
        
        success = process_single_image(
            image_path=image_file,
            text_prompt=text_prompt,
            config=config,
            output_dirs=output_dirs,
            detector=detector,
            sam_processor=sam_processor
        )
        
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    # Resource cleanup
    sam_processor.cleanup()
    
    # Print summary
    print_processing_summary(success_count, failure_count, len(image_files), output_dirs)
    
    if success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()