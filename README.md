# Language Inpaint-Anything

AI tool for removing objects from images using natural language descriptions

## Results Example

<div align="center">

| Original Image | Removal Result |
|----------------|----------------|
| ![Original](doc/original.jpg) | ![Result](doc/result.png) |

Prompt: `"red pylon"` → Red pylon is automatically removed and naturally restored

</div>

## Installation

```bash
# 1. Initialize submodules
git submodule update --init --recursive

# 2. Download model files
chmod +x install/download_models.sh
./install/download_models.sh

# 3. Install dependencies
pip install -r install/requirements.txt
pip install groundingdino-py
```

## Usage

```bash
# Run (using config.yaml)
python3 lang_inpaint_anything.py
```

### Output Structure

Processing results are saved in the following directory structure:

```
output_image/
├── GroundingDINO/    # Object detection result visualization
│   ├── image1.jpg
│   └── image2.jpg
├── SAM/              # Segmentation masks
│   ├── image1_mask0.jpg
│   └── image2_mask0.jpg
└── LAMA/             # Final restoration results
    ├── image1.jpg
    └── image2.jpg
```

## Key Features

- **Batch Processing Support**: Automatically processes all images in a directory
- **Separate Result Output**: Detection, segmentation, and restoration results saved separately
- **Memory Optimization**: GPU/CPU automatic switching, image resizing when memory is insufficient
- **Visualization**: Saves GroundingDINO detection results as images

## System Architecture

- **GroundingDINO**: Text-based object detection
- **SAM**: High-precision segmentation  
- **LaMa**: Natural image inpainting

## Configuration (config.yaml)

```yaml
# Processing parameters
processing:
  input_dir: "./input_image"       # Input image directory
  output_dir: "./output_image"     # Output base directory
  box_threshold: 0.35              # Object detection confidence
  dilate_kernel_size: 20           # Mask dilation size
  max_image_size: 2048             # Image resize limit for memory saving
  batch_processing: true           # Enable batch processing
  save_intermediate: true          # Save intermediate results

# Default values
general:
  default_text_prompt: "red pylon"
  output_subdirs:
    grounding_dino: "GroundingDINO"
    sam: "SAM"
    lama: "LAMA"

device:
  type: "auto"                     # cuda/cpu/auto
```

## Project Structure

```
├── lang_inpaint_anything.py     # Main script (batch processing support)
├── config.yaml                  # Configuration file
├── utils/                       # Utility modules
│   ├── config_loader.py         # Configuration management
│   ├── text_grounding_detector.py  # GroundingDINO wrapper
│   └── sam_processor.py         # SAM wrapper
├── input_image/                 # Input image directory
├── output_image/                # Result output
│   ├── GroundingDINO/          # Detection result visualization
│   ├── SAM/                    # Segmentation masks
│   └── LAMA/                   # Final restoration results
├── Inpaint-Anything/            # Submodule (SAM + LaMa)
└── doc/                         # Documentation and sample images
```

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended, also works with CPU)
- 8GB+ RAM

## References

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAM](https://github.com/facebookresearch/segment-anything)  
- [LaMa](https://github.com/advimman/lama)