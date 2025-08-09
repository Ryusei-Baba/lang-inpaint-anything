"""
Simple SAM processing module
Generates masks with SAM from bounding boxes detected by GroundingDINO
"""

import torch
import numpy as np
import cv2
import sys
import gc
from pathlib import Path
from typing import List, Tuple, Optional

# Add Inpaint-Anything path
sys.path.insert(0, str(Path(__file__).parent.parent / "Inpaint-Anything"))

try:
    from sam_segment import predict_masks_with_sam
    from utils import load_img_to_array, save_array_to_img, dilate_mask, show_mask
    SAM_AVAILABLE = True
except ImportError as e:
    print(f"SAM-related import error: {e}")
    SAM_AVAILABLE = False


class SAMProcessor:
    """Simple SAM segmentation processing class"""
    
    def __init__(self, 
                 model_type: str = "vit_h",
                 checkpoint_path: str = None,
                 device: str = "cuda"):
        """
        Initialize SAM processor
        
        Args:
            model_type: SAM model type (vit_h, vit_l, vit_b)
            checkpoint_path: SAM checkpoint file path
            device: Device to use (cuda/cpu/auto)
        """
        if not SAM_AVAILABLE:
            raise ImportError("SAM-related modules cannot be imported. Please check Inpaint-Anything configuration.")
        
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        
        # Device configuration
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"SAM Processor initialized: {model_type} on {self.device}")
    
    def boxes_to_points(self, boxes: torch.Tensor) -> Tuple[List[List[float]], List[int]]:
        """
        Convert bounding boxes to point prompts for SAM
        
        Args:
            boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
            
        Returns:
            points: List of point coordinates [[x, y], ...]
            labels: List of point labels [1, 1, ...]
        """
        points = []
        labels = []
        
        for box in boxes:
            # Get box center point
            x1, y1, x2, y2 = box.cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            points.append([float(center_x), float(center_y)])
            labels.append(1)  # Foreground point
        
        return points, labels
    
    def predict_masks(self, 
                     image: np.ndarray,
                     boxes: torch.Tensor) -> List[np.ndarray]:
        """
        Predict masks from bounding boxes
        
        Args:
            image: Input image (H, W, C) in RGB format
            boxes: Bounding boxes [N, 4]
            
        Returns:
            masks: List of predicted masks
        """
        if len(boxes) == 0:
            return []
        
        try:
            # Convert boxes to point prompts
            points, labels = self.boxes_to_points(boxes)
            
            print(f"Predicting masks with SAM... Point count: {len(points)}")
            
            # Execute SAM mask prediction
            masks, scores, logits = predict_masks_with_sam(
                img=image,
                point_coords=points,
                point_labels=labels,
                model_type=self.model_type,
                ckpt_p=self.checkpoint_path,
                device=self.device,
            )
            
            # Process results - select only the best mask
            if masks is not None and scores is not None:
                if isinstance(masks, np.ndarray):
                    # For multiple masks, select the highest scoring one
                    if len(masks.shape) == 3:
                        best_mask_idx = np.argmax(scores)
                        best_mask = masks[best_mask_idx]
                        print(f"Selected optimal mask (score: {scores[best_mask_idx]:.3f})")
                        return [best_mask]
                    # For single mask
                    elif len(masks.shape) == 2:
                        return [masks]
                elif isinstance(masks, list):
                    # For list case, also select highest scoring one
                    if len(masks) > 1 and scores is not None:
                        best_mask_idx = np.argmax(scores)
                        print(f"Selected optimal mask (score: {scores[best_mask_idx]:.3f})")
                        return [masks[best_mask_idx]]
                    return [masks[0]] if masks else []
            
            print("SAM mask prediction failed")
            return []
            
        except torch.cuda.OutOfMemoryError:
            print("GPU memory insufficient. Retrying with CPU...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Retry with CPU
            masks, scores, logits = predict_masks_with_sam(
                img=image,
                point_coords=points,
                point_labels=labels,
                model_type=self.model_type,
                ckpt_p=self.checkpoint_path,
                device="cpu",
            )
            
            if masks is not None and scores is not None:
                if isinstance(masks, np.ndarray):
                    # For multiple masks, select the highest scoring one
                    if len(masks.shape) == 3:
                        best_mask_idx = np.argmax(scores)
                        best_mask = masks[best_mask_idx]
                        print(f"CPU processing: Selected optimal mask (score: {scores[best_mask_idx]:.3f})")
                        return [best_mask]
                    # For single mask
                    elif len(masks.shape) == 2:
                        return [masks]
                elif isinstance(masks, list):
                    # For list case, also select highest scoring one
                    if len(masks) > 1 and scores is not None:
                        best_mask_idx = np.argmax(scores)
                        print(f"CPU processing: Selected optimal mask (score: {scores[best_mask_idx]:.3f})")
                        return [masks[best_mask_idx]]
                    return [masks[0]] if masks else []
            
            return []
            
        except Exception as e:
            print(f"Error occurred during SAM mask prediction: {e}")
            return []
    
    def dilate_masks(self, masks: List[np.ndarray], kernel_size: int = 15) -> List[np.ndarray]:
        """
        Dilate masks for expansion
        
        Args:
            masks: List of masks
            kernel_size: Dilation kernel size
            
        Returns:
            dilated_masks: List of masks after dilation
        """
        dilated_masks = []
        
        for mask in masks:
            try:
                # Use dilate_mask function
                dilated_mask = dilate_mask(mask, kernel_size)
                dilated_masks.append(dilated_mask)
            except Exception as e:
                print(f"Error during mask dilation: {e}")
                # Use original mask in case of error
                dilated_masks.append(mask)
        
        return dilated_masks
    
    def combine_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        """
        Combine multiple masks into one
        
        Args:
            masks: List of masks
            
        Returns:
            combined_mask: Combined mask
        """
        if len(masks) == 0:
            return None
        
        if len(masks) == 1:
            return masks[0]
        
        # Start with the first mask
        combined_mask = masks[0].copy()
        
        # Sequentially combine other masks
        for mask in masks[1:]:
            combined_mask = np.logical_or(combined_mask, mask)
        
        return combined_mask.astype(np.uint8)
    
    def cleanup(self):
        """Clean up resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def create_sam_processor(model_type: str = "vit_h",
                        checkpoint_path: str = None,
                        device: str = "cuda") -> SAMProcessor:
    """
    SAM processor factory function
    
    Args:
        model_type: SAM model type
        checkpoint_path: Checkpoint file path
        device: Device to use
        
    Returns:
        SAMProcessor instance
    """
    return SAMProcessor(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        device=device
    )