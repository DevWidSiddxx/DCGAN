# DCGAN for Medical Image Generation

## Overview
This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate synthetic medical images, specifically for diabetic foot ulcer (DFU) datasets. The implementation includes data loading, model architecture, training loop, and output generation.

## Features
- Custom dataset loader for medical images
- DCGAN generator and discriminator architectures
- Training loop with progress monitoring
- Model checkpointing and sample generation
- Google Drive integration for data storage

## Requirements
- Python 3.x
- PyTorch (>=1.8.0)
- Torchvision
- Pillow (PIL)
- Google Colab (for cloud execution)
- NVIDIA GPU (recommended)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-dcgan.git
cd medical-dcgan
```

2. Install dependencies:
```bash
pip install torch torchvision pillow
```

## Usage

### Google Colab Setup
1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Set up your data directory path in the notebook.

### Training the Model
The main training loop is in the Jupyter notebook. Key parameters you can adjust:
```python
# Hyperparameters
batch_size = 64
image_size = 64
nz = 100  # Size of latent vector
ngf = 64  # Generator feature maps
ndf = 64  # Discriminator feature maps
nc = 3    # Number of image channels
num_epochs = 50
```

### Monitoring Progress
The training outputs:
- Loss values for generator and discriminator
- Sample generated images every epoch
- Model checkpoints saved periodically

## File Structure
```
medical-dcgan/
├── DCGAN.ipynb               # Main Jupyter notebook
├── MedicalImageDataset.py    # Custom dataset loader
├── Generator.py             # Generator network
├── Discriminator.py         # Discriminator network
├── samples/                 # Generated image samples
└── models/                  # Saved model checkpoints
```

## Results
After training, you can expect:
- Generated medical images in the output directory
- Generator and discriminator model checkpoints
- Training statistics showing the adversarial balance

## Customization
To adapt this for your own dataset:
1. Place your images in a directory
2. Update the `root_dir` path in the `MedicalImageDataset` initialization
3. Adjust image size parameters if needed

## Troubleshooting
- If you get CUDA memory errors, reduce the batch size
- For slow training, try smaller network sizes (ngf, ndf)
- Ensure all images are the same size and format (JPEG/PNG)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
