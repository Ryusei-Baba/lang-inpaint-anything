#!/bin/bash

# Download models for Text Remove Anything
echo "=== Downloading models for Text Remove Anything ==="

# Create directories
mkdir -p Inpaint-Anything/pretrained_models
mkdir -p Inpaint-Anything/weights

echo "Downloading SAM model..."
# Download SAM ViT-H model
wget -O Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

echo "Downloading MobileSAM model..."
# Download MobileSAM model  
wget -O Inpaint-Anything/weights/mobile_sam.pt \
    https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt?raw=true

echo "Downloading LaMa model..."
# Create LaMa directory and download
mkdir -p Inpaint-Anything/pretrained_models/big-lama
cd Inpaint-Anything/pretrained_models/big-lama

# Download LaMa model files (using working Hugging Face URL)
wget https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip
rm big-lama.zip

cd ../../../

echo "Downloading GroundingDINO model..."
# Create GroundingDINO directory and download
mkdir -p models/grounding_dino

# Download GroundingDINO model
wget -O models/grounding_dino/groundingdino_swint_ogc.pth \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download config file
wget -O models/grounding_dino/GroundingDINO_SwinT_OGC.py \
    https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py

echo "=== Model download complete! ==="
echo ""
echo "Downloaded models:"
echo "- SAM ViT-H: Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth"
echo "- MobileSAM: Inpaint-Anything/weights/mobile_sam.pt"  
echo "- LaMa: Inpaint-Anything/pretrained_models/big-lama/"
echo "- GroundingDINO: models/grounding_dino/groundingdino_swint_ogc.pth"
echo ""
echo "Next steps:"
echo "1. Install Python dependencies:"
echo "   pip install -r install/requirements.txt"
echo "   pip install groundingdino-py"
echo ""
echo "You can now run Text Remove Anything!"
echo ""