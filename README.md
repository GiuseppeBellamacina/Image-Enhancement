# 📸 Image Enhancement with Deep Learning

## Corruption → Restoration → Evaluation

A comparative study of **eight deep learning architectures** for image denoising, developed as a Deep Learning course project. The system applies controlled degradations to high-resolution images and restores them using CNN-based models, evaluating results with quantitative metrics.

**Pipeline:**  
👉 Corrupt images with parametric degradations  
👉 Restore them using advanced CNN models  
👉 Compare results with quantitative (PSNR/SSIM) and qualitative metrics

📄 **[Full Report (PDF)](docs/latex/relazione.pdf)** — IEEE-format paper with detailed architecture descriptions, training procedures, ablation studies, and results analysis.

---

## 🏆 Results Summary

| Model          | PSNR (dB) | SSIM      | Parameters |
| -------------- | --------- | --------- | ---------- |
| **NAFNet-V2**  | **26.36** | **0.747** | ~67M       |
| Pix2Pix        | 26.08     | 0.733     | ~54M       |
| RIDNet         | 25.83     | 0.720     | ~1.5M      |
| DnCNN          | 25.62     | 0.710     | ~0.7M      |
| Attention UNet | 25.55     | 0.709     | ~13.7M     |
| UNet Residual  | 25.45     | 0.697     | ~7.8M      |
| UNet           | 25.17     | 0.686     | ~7.8M      |
| Denoising AE   | 24.46     | 0.651     | ~2.3M      |

Evaluated on DIV2K validation images with Gaussian noise σ=100.

---

## 🔍 Project Goals

- [x] Implement **parametric degradation** types (Gaussian noise, quantization dithering, salt & pepper)
- [x] **Automatic path management** for degraded datasets
- [x] Train **8 CNN restoration models** (UNet, UNet Residual, Attention UNet, DnCNN, RIDNet, DAE, Pix2Pix, NAFNet-V2)
- [x] Implement **advanced loss functions** (L1+SSIM, Perceptual Loss with VGG16, Charbonnier, adversarial)
- [x] Evaluate with **PSNR, SSIM** metrics + sliding window inference on full-resolution images
- [x] Complete **training system** (mixed precision, warmup, cosine scheduler, early stopping)
- [x] **Ablation studies** on architectures, loss functions, and training strategies

---

## 📁 Repository Structure

```text
├── 📁 data/                    # Datasets (raw + degraded)
├── 📁 docs/latex/              # LaTeX report source
├── 📁 experiments/             # Training outputs and checkpoints
├── 📁 notebooks/
│   ├── 📁 test_degradations/   # Degradation testing notebooks
│   └── 📁 test_trainings/      # Training notebooks per model
├── 📁 report_generators/       # Scripts for generating reports
├── 📁 src/
│   ├── 📁 degradations/        # Image corruption implementations
│   ├── 📁 evaluation/          # Metrics and inference
│   ├── 📁 losses/              # Loss functions
│   ├── 📁 models/              # Model architectures
│   ├── 📁 training/            # Training pipeline
│   └── 📁 utils/               # Utilities (paths, checkpoints, etc.)
├── 📄 requirements.txt
├── 📄 setup.ps1                # Windows setup script
└── 📄 pyproject.toml
```

---

## 🧪 Implemented Degradations

### Gaussian Noise

Additive Gaussian noise with configurable sigma.

- Parameterized by σ (standard deviation)
- Training tested at σ = 100 (heavy noise)
- Auto-path: `data/degraded/gaussian/sigma_{sigma}/`

### Quantization + Dithering

Color quantization with variable bit depth + dithering algorithms.

- Quantization: 2, 4, 6, 8 bits per channel
- Dithering: random, Floyd-Steinberg, Bayer pattern
- Auto-path: `data/degraded/dithering/{type}/{bits}bit/`

### Salt & Pepper Noise

- Variable density
- Configurable salt/pepper ratio

---

## 🤖 Implemented Models

| Model              | Type            | Key Features                                   |
| ------------------ | --------------- | ---------------------------------------------- |
| **UNet**           | Encoder-Decoder | Skip connections, direct reconstruction        |
| **UNet Residual**  | Encoder-Decoder | Residual learning (predicts noise)             |
| **Attention UNet** | Encoder-Decoder | Attention gates on skip connections            |
| **DnCNN**          | Feed-Forward    | 17-layer CNN with batch normalization          |
| **RIDNet**         | Residual        | Feature attention + EAM modules                |
| **Denoising AE**   | Autoencoder     | Compact encoder-decoder                        |
| **Pix2Pix**        | GAN             | Conditional adversarial training               |
| **NAFNet-V2**      | Activation-Free | Simple Channel Attention, multi-stage training |

---

## 🎯 Loss Functions

| Loss                       | Formula                   | Use Case                       |
| -------------------------- | ------------------------- | ------------------------------ |
| **CombinedLoss**           | α·L1 + β·(1-SSIM)         | General denoising              |
| **CombinedPerceptualLoss** | α·L1 + β·(1-SSIM) + γ·VGG | Texture preservation           |
| **Charbonnier**            | √(x² + ε²)                | Smooth L1 alternative (NAFNet) |
| **RIDNet Loss**            | L1 + edge-aware           | Feature attention training     |
| **Adversarial**            | GAN loss + λ·L1           | Pix2Pix training               |

---

## 📊 Evaluation

- **PSNR** (Peak Signal-to-Noise Ratio) — pixel-wise quality (dB)
- **SSIM** (Structural Similarity Index) — structural similarity (0-1)
- **Sliding window inference** — full-resolution evaluation with Hann window blending
- Patch size: 128×128 with 32px overlap

---

## 🚀 Training System Features

- **Mixed Precision (AMP)**: faster training with FP16/FP32
- **Warmup scheduling**: linear warmup + cosine annealing
- **Early stopping**: patience-based with best model tracking
- **Gradient clipping**: max_norm=1.0 for stability
- **Checkpointing**: best model + periodic saves
- **Experiment management**: auto-naming, config saving, TensorBoard logging
- **Telegram notifications**: automatic progress updates
- **Data augmentation**: random crops, flips, normalization

---

## 🚀 Setup

### Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 8GB+ RAM
- 4GB+ VRAM (recommended for batch_size=16)

### Installation

```bash
# Clone repository
git clone https://github.com/GiuseppeBellamacina/Image-Enhancement.git
cd Image-Enhancement

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Or use setup script (Windows + uv)
.\setup.ps1
```

### Key Dependencies

- `torch>=2.0.0` — PyTorch framework
- `torchvision>=0.15.0` — Pre-trained models and transforms
- `pytorch-msssim` — Differentiable SSIM loss
- `opencv-python` — Image I/O and processing
- `scikit-image` — Metrics (PSNR, SSIM)
- `tensorboard` — Experiment logging

---

## 📊 Dataset

### DIV2K

[**DIV2K**](https://data.vision.ee.ethz.ch/cvl/DIV2K/) — High-quality 2K resolution dataset for image restoration.

- 800 training images + 100 validation images
- Automatically downloaded via `download_div2k_dataset()`

```text
data/raw/
├── DIV2K_train_HR/  # 800 images
└── DIV2K_valid_HR/  # 100 images
```

---

## 📖 Usage

### 1. Download Dataset

```python
from src.utils import download_div2k_dataset
download_div2k_dataset()
```

### 2. Generate Degraded Dataset

```python
from src.degradations import generate_degraded_dataset_auto

train_deg, train_clean = generate_degraded_dataset_auto(
    dataset_split="DIV2K_train_HR",
    degradation_type="gaussian_noise",
    noise_sigma=100.0,
    seed=42
)
```

### 3. Training

See notebooks in `notebooks/test_trainings/` for complete examples.

```python
from src.models import UNet
from src.losses import CombinedLoss
from src.training import get_dataloaders, run_training

model = UNet(in_channels=3, out_channels=3, features=64, bilinear=True)
criterion = CombinedLoss(alpha=0.84, beta=0.16)

train_loader, val_loader = get_dataloaders(
    train_degraded_dir="data/degraded/gaussian/sigma_100/DIV2K_train_HR",
    train_clean_dir="data/raw/DIV2K_train_HR",
    val_degraded_dir="data/degraded/gaussian/sigma_100/DIV2K_valid_HR",
    val_clean_dir="data/raw/DIV2K_valid_HR",
    batch_size=16, patch_size=128, patches_per_image=20
)

history, best_info = run_training(
    model=model, train_loader=train_loader, val_loader=val_loader,
    criterion=criterion, optimizer=optimizer, device="cuda",
    num_epochs=36, use_amp=True
)
```

### 4. Evaluation

```python
from src.evaluation import ImageRestorationEvaluator

evaluator = ImageRestorationEvaluator(model=model, device="cuda", patch_size=128, overlap=32)
results = evaluator.evaluate_dataset(
    degraded_dir="data/degraded/gaussian/sigma_100/DIV2K_valid_HR",
    clean_dir="data/raw/DIV2K_valid_HR",
    output_dir="experiments/unet/gaussian/restored_images",
    save_outputs=True
)
evaluator.print_summary(results)
```

### 5. TensorBoard Monitoring

```bash
tensorboard --logdir experiments/
```

---

## 👥 Team

| Member                   | Contributions                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| **Giuseppe Bellamacina** | UNet, UNet Residual, Attention UNet, Loss Functions, Training System, Evaluation, Infrastructure |
| **Daniele Barbagallo**   | Pix2Pix GAN, NAFNet-V2                                                                           |
| **Salvatore Iurato**     | DnCNN, RIDNet                                                                                    |
| **Mattia Campanella**    | Denoising Autoencoder                                                                            |

---

## 📚 References

**Architectures:**

- **UNet**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **Attention UNet**: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas" (2018)
- **DnCNN**: Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising" (2017)
- **RIDNet**: Anwar & Barnes, "Real Image Denoising with Feature Attention" (2019)
- **NAFNet**: Chen et al., "Simple Baselines for Image Restoration" (2022)
- **Pix2Pix**: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (2017)

**Loss Functions:**

- **SSIM**: Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (2004)
- **Perceptual Loss**: Johnson et al., "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (2016)

**Dataset:**

- **DIV2K**: Agustsson & Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution" (2017)

---

## 👥 Authors

### Giuseppe Bellamacina

### Daniele Barbagallo

### Salvatore Iurato

### Mattia Campanella

Progetto sviluppato per il corso di **Deep Learning** — A.A. 2025/2026

---

## 📎 License

MIT License

Copyright (c) 2025 Giuseppe Bellamacina, Daniele Barbagallo, Salvatore Iurato, Mattia Campanella

---

## 🙏 Acknowledgments

- PyTorch team per il framework
- DIV2K dataset creators
- pytorch-msssim library per SSIM differenziabile
- VS Code + Copilot per development support

---

**Note:** Progetto in sviluppo attivo. README aggiornato regolarmente con nuove features e risultati.
