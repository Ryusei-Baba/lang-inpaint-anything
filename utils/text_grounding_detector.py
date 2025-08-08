import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util import box_ops
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    GROUNDINGDINO_AVAILABLE = True
except ImportError:
    GROUNDINGDINO_AVAILABLE = False
    print("Warning: GroundingDINO not available. Please install it first.")

def load_image(image_path: str) -> Tuple[np.ndarray, torch.Tensor]:
    """Load and preprocess image for GroundingDINO"""
    image_pil = cv2.imread(image_path)
    image_pil = cv2.cvtColor(image_pil, cv2.COLOR_BGR2RGB)
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor, _ = transform(image_pil, None)
    return image_pil, image_tensor

def load_model_hf(model_config_path: str, repo_id: str, filename: str, device: str = 'cpu'):
    """Load GroundingDINO model from HuggingFace"""
    try:
        from huggingface_hub import hf_hub_download
        
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        config_file = hf_hub_download(repo_id=repo_id, filename="GroundingDINO_SwinT_OGC.py")
        
        args = SLConfig.fromfile(config_file)
        model = build_model(args)
        checkpoint = torch.load(cache_file, map_location=device)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        raise

class TextGroundingDetector:
    """Text-based object detection using GroundingDINO"""
    
    def __init__(self, 
                 config_path: str = None,
                 checkpoint_path: str = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        if not GROUNDINGDINO_AVAILABLE:
            raise ImportError("GroundingDINO is not available. Please install it first.")
        
        self.device = device
        
        # Use HuggingFace model if no local paths provided
        if config_path is None or checkpoint_path is None:
            try:
                print("Loading GroundingDINO from HuggingFace...")
                repo_id = "IDEA-Research/grounding-dino"
                filename = "groundingdino_swint_ogc.pth"
                
                self.model = load_model_hf(config_path, repo_id, filename, device)
            except Exception as e:
                print(f"Failed to load from HuggingFace: {e}")
                print("Please provide local config_path and checkpoint_path, or install GroundingDINO locally.")
                raise ImportError("GroundingDINO model could not be loaded. Please check your installation or provide local model files.")
        else:
            # Load from local files
            args = SLConfig.fromfile(config_path)
            self.model = build_model(args)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            self.model.eval()
        
        self.model.to(device)
        
    def predict(self, 
                image: np.ndarray,
                text_prompt: str,
                box_threshold: float = 0.35,
                text_threshold: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Predict objects in image based on text prompt
        
        Args:
            image: Input image as numpy array
            text_prompt: Text description of objects to detect
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
            
        Returns:
            boxes: Bounding boxes in format (x1, y1, x2, y2)
            logits: Confidence scores
            phrases: Detected phrases
        """
        
        # Convert numpy array to PIL Image
        from PIL import Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype(np.uint8))
        else:
            image_pil = image
        
        # Preprocess image
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor, _ = transform(image_pil, None)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Ensure text prompt ends with period for GroundingDINO
        if not text_prompt.endswith('.'):
            text_prompt = text_prompt + '.'
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor, captions=[text_prompt])
        
        # Process outputs
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # Filter by thresholds
        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # (n, 256)
        boxes = prediction_boxes[mask]  # (n, 4)
        
        # Get phrases
        tokenlizer = self.model.tokenizer
        tokenized = tokenlizer(text_prompt)
        phrases = []
        
        for logit in logits:
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, 
                                                 tokenized, 
                                                 tokenlizer)
            phrases.append(pred_phrase)
        
        # Convert boxes to absolute coordinates
        if isinstance(image, np.ndarray):
            H, W = image.shape[:2]
        else:
            W, H = image.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * torch.tensor([W, H, W, H])
        
        return boxes, logits.max(dim=1)[0], phrases
    
    def detect_objects(self, 
                      image_path: str,
                      text_prompt: str,
                      box_threshold: float = 0.35,
                      text_threshold: float = 0.25) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, List[str]]:
        """
        Detect objects in image file
        
        Args:
            image_path: Path to input image
            text_prompt: Text description of objects to detect
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
            
        Returns:
            image: Original image as numpy array
            boxes: Bounding boxes
            logits: Confidence scores  
            phrases: Detected phrases
        """
        
        # Load image with OpenCV and convert to RGB
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes, logits, phrases = self.predict(image, text_prompt, box_threshold, text_threshold)
        
        return image, boxes, logits, phrases

def boxes_to_sam_prompts(boxes: torch.Tensor, format: str = "box") -> List:
    """
    Convert GroundingDINO boxes to SAM prompts
    
    Args:
        boxes: Bounding boxes from GroundingDINO
        format: Format for SAM prompts ("box" or "point")
        
    Returns:
        List of SAM prompts
    """
    if format == "box":
        # Return boxes as-is for SAM box prompts
        return boxes.numpy().tolist()
    elif format == "point":
        # Convert boxes to center points
        points = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            points.append([center_x.item(), center_y.item()])
        return points
    else:
        raise ValueError(f"Unsupported format: {format}")

def visualize_detections(image: np.ndarray, 
                        boxes: torch.Tensor, 
                        phrases: List[str],
                        logits: torch.Tensor,
                        save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize detection results
    
    Args:
        image: Input image
        boxes: Bounding boxes
        phrases: Detected phrases
        logits: Confidence scores
        save_path: Path to save visualization
        
    Returns:
        Annotated image
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw bounding boxes
    for box, phrase, confidence in zip(boxes, phrases, logits):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Draw rectangle
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add text
        ax.text(x1, y1 - 10, f"{phrase}: {confidence:.2f}", 
                fontsize=12, color='red', weight='bold')
    
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close()
    
    return image