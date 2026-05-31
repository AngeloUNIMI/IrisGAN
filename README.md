<div align="center">

# 👁️ IrisGAN

### Iris deidentification with high visual realism for privacy protection on websites and social networks

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GAN-red?logo=pytorch)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-00629B)](https://ieeexplore.ieee.org/document/9543669)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://iebil.di.unimi.it/irisGan/irisGan.html)

**PyTorch source code for the IEEE Access 2021 paper**  
*Iris deidentification with high visual realism for privacy protection on websites and social networks*

</div>

---

## 🧠 Overview

**IrisGAN** is a PyTorch implementation of a Generative Adversarial Network designed for **iris deidentification**.

The goal is to generate realistic iris textures that can replace the biometric information contained in the iris region while preserving visual plausibility. This helps reduce privacy risks caused by high-resolution face and eye images shared on websites, social networks, and other public platforms.

The method was proposed to address a concrete privacy issue: iris patterns visible in personal images can potentially be used for automatic biometric recognition. IrisGAN replaces identity-bearing iris texture information with synthetic, realistic content.

---

## ✨ Key Features

- 👁️ **Iris deidentification** using synthetic texture generation
- 🧬 **GAN-based image synthesis** implemented in PyTorch
- 🖼️ Works on **Rubber Sheet Model** iris representations
- ⚙️ Training and generation scripts included
- 💾 Saves generated iris textures and trained generator models
- 🔬 Designed for research on biometric privacy protection

---

## 🧩 Method at a Glance

<div align="center">

![IrisGAN outline](https://iebil.di.unimi.it/irisGan/imgs/outline.jpg "IrisGAN outline")

</div>

```text
Rubber Sheet Model iris textures
              │
              ▼
      GAN training stage
              │
      ┌───────┴────────┐
      ▼                ▼
 Generator       Discriminator
      │                ▲
      └──── synthetic iris feedback
              │
              ▼
Synthetic realistic iris textures
              │
              ▼
Iris replacement for deidentification
```

---

## 📁 Repository Structure

```text
IrisGAN/
│
├── DCGAN-PyTorch_A_train.py     # Train the GAN on iris RSM images
├── DCGAN-PyTorch_B_test.py      # Load a trained generator and synthesize iris textures
├── rsm/
│   └── 1/                       # Input Rubber Sheet Model iris images
├── images/                      # Generated samples during training
├── images_generated/            # Synthetic iris textures generated at test time
├── models/                      # Saved generator models
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AngeloUNIMI/IrisGAN.git
cd IrisGAN
```

### 2. Install dependencies

The code requires Python and PyTorch. A typical setup is:

```bash
pip install torch torchvision numpy pillow
```

A CUDA-capable GPU is recommended for training, but the scripts can also run on CPU if CUDA is not available.

---

## 🗂️ Dataset Preparation

Place the Rubber Sheet Model iris images in:

```text
./rsm/1/
```

The expected input format is:

| Property | Value |
|---|---|
| Representation | Rubber Sheet Model iris texture |
| Size | `512 × 64` |
| Color depth | 8-bit grayscale |
| Channels | 1 |

The repository already includes some example files.

---

## 🏋️ Training

Run the training script:

```bash
python DCGAN-PyTorch_A_train.py
```

You can customize the main training parameters from the command line:

```bash
python DCGAN-PyTorch_A_train.py \
  --n_epochs 200 \
  --batch_size 60 \
  --lr 0.0002 \
  --latent_dim 100 \
  --img_sizeH 64 \
  --img_sizeW 512 \
  --channels 1
```

During training, the script saves:

```text
./images/       # Generated sample images
./models/       # Saved generator checkpoints
```

Model files are saved with names such as:

```text
model_save_20000.pth
```

---

## 🎨 Generating Synthetic Iris Textures

After training, generate synthetic iris textures using:

```bash
python DCGAN-PyTorch_B_test.py --batches 20000
```

You can also specify how many images to generate:

```bash
python DCGAN-PyTorch_B_test.py \
  --batches 20000 \
  --n_images 1000 \
  --batch_size 2
```

Generated images are saved in:

```text
./images_generated/
```

---

## 📊 Inputs and Outputs

| Stage | Input | Output |
|---|---|---|
| Training | Real RSM iris textures in `./rsm/1/` | GAN samples in `./images/` and models in `./models/` |
| Generation | Saved model in `./models/` | Synthetic iris textures in `./images_generated/` |

---

## 🔬 Research Context

IrisGAN was developed to support privacy protection in scenarios where high-resolution facial or ocular images may reveal biometric iris information. The original paper shows that iris information visible in images from the web and social media can pose a recognition risk, and proposes GAN-based iris replacement as a deidentification strategy.

---

## 📖 Paper

If you use this code, please cite:

```bibtex
@Article{iride21,
  author  = {M. Barni and R. {Donida Labati} and A. Genovese and V. Piuri and F. Scotti},
  title   = {Iris deidentification with high visual realism for privacy protection on websites and social networks},
  journal = {IEEE Access},
  volume  = {9},
  pages   = {131995--132010},
  year    = {2021},
  note    = {2169-3536}
}
```

**Publication**  
M. Barni, R. Donida Labati, A. Genovese, V. Piuri, and F. Scotti,  
*Iris deidentification with high visual realism for privacy protection on websites and social networks*,  
IEEE Access, vol. 9, pp. 131995–132010, 2021.  
DOI: [`10.1109/ACCESS.2021.3114588`](https://doi.org/10.1109/ACCESS.2021.3114588)

---

## 🔗 Links

| Resource | Link |
|---|---|
| Paper | https://ieeexplore.ieee.org/document/9543669 |
| DOI | https://doi.org/10.1109/ACCESS.2021.3114588 |
| Project page | https://iebil.di.unimi.it/irisGan/irisGan.html |
| Source code | https://github.com/AngeloUNIMI/IrisGAN |

---

## 👥 Authors

- **Mauro Barni**  
  Department of Information Engineering and Mathematics, Università degli Studi di Siena, Italy

- **Ruggero Donida Labati**  
  Department of Computer Science, Università degli Studi di Milano, Italy

- **Angelo Genovese**  
  Department of Computer Science, Università degli Studi di Milano, Italy

- **Vincenzo Piuri**  
  Department of Computer Science, Università degli Studi di Milano, Italy

- **Fabio Scotti**  
  Department of Computer Science, Università degli Studi di Milano, Italy

---

## 📄 License

This project is released under the **GNU General Public License v3.0**.

See the [LICENSE](LICENSE) file for details.
