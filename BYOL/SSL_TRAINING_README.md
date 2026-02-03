# Solar Panel SSL Training - BYOL

## Overview
This script implements BYOL (Bootstrap Your Own Latent) for self-supervised learning on solar panel imagery.

## Dataset
- **Path**: `/home/craib/solar-segmentation-annotation-ssl/solar-dataset`
- **Images**: 836 PNG files
- **Format**: RGB satellite imagery

## Installation

```bash
# Install required packages
pip install -r requirements.txt
```

## Configuration

The script is configured with the following paths:
- **Dataset**: `/home/craib/solar-segmentation-annotation-ssl/solar-dataset`
- **Output**: `/home/craib/solar-segmentation-annotation-ssl/Shahzeb/outputs`

## Running the Training

```bash
cd /home/craib/solar-segmentation-annotation-ssl/Shahzeb
python ssl_solar_v3.py
```

## Training Parameters

- **Method**: BYOL (Bootstrap Your Own Latent)
- **Backbone**: ResNet50 (ImageNet pretrained)
- **Epochs**: 200
- **Batch Size**: 128 (effective 512 with 4x gradient accumulation)
- **Base Learning Rate**: 0.2 (with LARS optimizer)
- **Warmup Epochs**: 10

## Output Files

All outputs are saved to `outputs/` directory:
- `byol_best.pth` - Best model checkpoint
- `byol_encoder_final.pth` - Final encoder weights
- `byol_training_log.csv` - Training metrics log
- `byol_epoch_*.pth` - Checkpoints every 20 epochs

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (script will use CPU if GPU unavailable)
- **RAM**: At least 8GB
- **Storage**: ~2GB for model checkpoints

## Using the Pretrained Encoder

After training, load the encoder for downstream segmentation tasks:

```python
from ssl_solar_v3 import load_pretrained_backbone

# Load the pretrained encoder
backbone = load_pretrained_backbone("outputs/byol_best.pth")

# Use with your segmentation model (DeepLabV3, U-Net, etc.)
```

## Monitoring Training

The script outputs:
- Loss values per epoch
- Feature standard deviation (diversity metric)
- Learning rate schedule
- Momentum schedule

**Health indicators**:
- ✅ Feature std > 0.1: Good diversity
- ⚠️ Feature std 0.05-0.1: Low diversity
- ❌ Feature std < 0.05: Collapse detected
