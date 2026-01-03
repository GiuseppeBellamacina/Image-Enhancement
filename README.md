# 📸 Image Enhancement with Deep Learning

**Corruption → Restoration → Evaluation**

Progetto di studio e confronto di diversi metodi di **Image Enhancement** attraverso reti neurali convoluzionali e tecniche di degradazione controllata.

**Pipeline:**  
👉 Corrompere immagini con degradazioni parametrizzabili  
👉 Ricostruirle usando modelli CNN avanzati (UNet, Residual UNet, Attention UNet)  
👉 Confrontare i risultati con metriche quantitative (PSNR/SSIM) e qualitative

---

## 🔍 Obiettivi del progetto

- [x] Implementare **tipi di corruzione** parametrizzabili (Gaussian noise, Quantization dithering)
- [x] Sistema di **path management automatico** per dataset degradati
- [x] Addestrare modelli di **restauro CNN** (UNet, UNetResidual, AttentionUNet)
- [x] Implementare **loss functions** avanzate (L1+SSIM, Perceptual Loss con VGG16)
- [x] Valutare con metriche **PSNR, SSIM** + sliding window inference su immagini full-resolution
- [x] Sistema di **training completo** (mixed precision, warmup, cosine scheduler, early stopping)
- [ ] Estendere a più degradazioni (blur, JPEG compression, low-light)
- [ ] Ablation study completo su architetture e loss functions

---

## 📁 Struttura della Repository

```
Image-Enhancement/
├── data/
│   ├── raw/                          # Dataset originale DIV2K
│   │   ├── DIV2K_train_HR/          # 800 immagini training HR
│   │   └── DIV2K_valid_HR/          # 100 immagini validation HR
│   └── degraded/                     # Immagini corrotte (auto-generato)
│       ├── gaussian/                 # Gaussian noise
│       │   └── sigma_100/           # σ=100
│       │       ├── DIV2K_train_HR/
│       │       └── DIV2K_valid_HR/
│       └── dithering/                # Quantization dithering
│           └── random/               # Random dithering
│               ├── 4bit/             # 4-bit quantization
│               │   ├── DIV2K_train_HR/
│               │   └── DIV2K_valid_HR/
│               └── 6bit/             # 6-bit quantization
│
├── src/
│   ├── degradations/                 # Script corruzioni immagini
│   │   ├── gaussian_noise.py        # Gaussian noise implementation
│   │   ├── quantization_dithering.py # Quantization + dithering
│   │   └── generate_degraded_dataset.py # Auto-path generation system
│   ├── models/                       # Architetture CNN
│   │   ├── unet.py                  # UNet standard
│   │   ├── unet_residual.py         # UNet con residual learning
│   │   └── attention_unet.py        # UNet con attention gates
│   ├── losses/                       # Funzioni di loss
│   │   ├── combined_loss.py         # L1 + SSIM
│   │   └── perceptual_loss.py       # L1 + SSIM + VGG Perceptual
│   ├── training/                     # Training pipeline
│   │   ├── dataset.py               # PatchDataset con augmentation
│   │   ├── training.py              # train_epoch + validate
│   │   └── trainer.py               # run_training con AMP, warmup, scheduler
│   ├── evaluation/                   # Valutazione modelli
│   │   ├── metrics.py               # PSNR, SSIM calculation
│   │   ├── inference.py             # Sliding window inference
│   │   └── evaluator.py             # ImageRestorationEvaluator
│   └── utils/                        # Utility functions
│       ├── checkpoints.py           # Gestione checkpoint
│       ├── experiment.py            # Setup esperimenti
│       ├── paths.py                 # Path management
│       ├── visualization.py         # Plot utilities
│       ├── download_dataset.py      # DIV2K downloader
│       └── telegram_notifier.py     # Notifiche Telegram
│
├── experiments/                      # Risultati training
│   ├── unet/                        # UNet experiments
│   │   ├── gaussian/                # Gaussian noise restoration
│   │   │   └── 20260103_135525_bilinear/
│   │   │       ├── config.json
│   │   │       ├── history.json
│   │   │       ├── evaluation_metrics.json
│   │   │       ├── checkpoints/     # best_model.pth, epoch_*.pth
│   │   │       ├── samples/         # Immagini di esempio
│   │   │       ├── restored_images/ # Full-res validation
│   │   │       └── logs/            # TensorBoard logs
│   │   └── dithering/               # Dithering restoration
│   ├── unet_residual/               # UNet Residual experiments
│   └── attention_unet/              # Attention UNet experiments
│
├── notebooks/                        # Jupyter notebooks
│   ├── test_degradations/           # Test degradation functions
│   │   ├── test_gaussian_noise.ipynb
│   │   └── test_quantization_dithering.ipynb
│   └── test_trainings/              # Training notebooks
│       ├── unet/
│       │   ├── gaussian/
│       │   │   └── unet_gaussian_bilinear.ipynb
│       │   └── dithering/
│       │       └── random/
│       ├── unet_residual/
│       │   └── gaussian/
│       │       ├── unet_residual_gaussian_bilinear.ipynb
│       │       └── unet_residual_gaussian_upsample.ipynb
│       └── attention_unet/
│           └── gaussian/
│               └── attention_unet_gaussian_bilinear.ipynb
│
├── requirements.txt                  # Dipendenze Python
├── pyproject.toml                   # Package configuration
├── setup.ps1                        # PowerShell setup script
├── format.ps1                       # Code formatting script
└── README.md
```

---

## 🧪 Degradazioni Implementate

### ✅ Gaussian Noise

Rumore gaussiano additivo parametrizzabile per sigma.

**Implementazione:**

- Noise parametrizzato da σ (standard deviation)
- Training testato: σ = 100 (noise pesante)
- Path automatico: `data/degraded/gaussian/sigma_{sigma}/`

### ✅ Quantization + Dithering

Quantizzazione del colore con diversi livelli di bit depth + dithering.

**Implementazione:**

- Quantizzazione: 2, 4, 6, 8 bit per canale
- Dithering: random, Floyd-Steinberg, Bayer pattern
- Path automatico: `data/degraded/dithering/{type}/{bits}bit/`

**Configurazione testata:**

- 2-bit random dithering
- Training: immagini ditherate → originali clean

### 🔜 Future Degradations (Planned)

- **Salt & Pepper noise** (densità variabile)
- **Gaussian blur** / **Motion blur** (kernel size variabile)
- **JPEG compression artifacts** (quality: 30, 50, 70, 90)
- **Low-light simulation** (gamma correction + scaling)
- **Combinazioni** (es. blur + noise, JPEG + dithering)

---

## 🤖 Modelli Implementati

### ✅ UNet (Standard)

Architettura encoder-decoder con skip connections.

**Caratteristiche:**

- Encoder: 4 livelli di downsampling (conv + max pool)
- Decoder: 4 livelli di upsampling (bilinear/transposed conv)
- Skip connections: concatenazione features encoder → decoder
- Output: Direct reconstruction (predice immagine pulita)

**Parametri:**

- features=64, bilinear=True: ~7.8M params
- features=64, bilinear=False: ~11M params

### ✅ UNet Residual

UNet con residual learning: predice noise invece di immagine.

**Caratteristiche:**

- Stessa architettura di UNet standard
- Output: `clean = degraded - predicted_noise`
- Migliore per denoising (apprende direttamente il rumore)

**Vantaggi:**

- Convergenza più veloce su Gaussian noise
- Gradients più stabili

### ✅ Attention UNet

UNet con attention gates per focus selettivo sulle regioni importanti.

**Caratteristiche:**

- Attention gates su ogni skip connection
- Focus automatico su regioni degradate
- Parametri: ~13.7M (bilinear=True), ~17-18M (bilinear=False)

**Configurazione ottimale:**

- bilinear=True per stabilità
- Learning rate: 3e-5 - 5e-5
- Weight decay: 1e-6

### 🔜 Future Models (Planned)

**CNN-based:**

- **DnCNN** (Denoising CNN con batch norm)
- **Denoising Autoencoder** (encoder-decoder semplice)

**Advanced:**

- **Transformer-based** (SwinIR, opzionale)
- **GAN-based** (Pix2Pix per texture enhancement)

---

## 🎯 Loss Functions Implementate

### ✅ CombinedLoss (L1 + SSIM)

Loss combination per bilanciare pixel-wise e structural similarity.

**Formula:**

```python
loss = α * L1(pred, target) + β * (1 - SSIM(pred, target))
```

**Configurazione tipica:**

- α = 0.84 (L1 weight)
- β = 0.16 (SSIM weight)

**Vantaggi:**

- L1: Convergenza pixel-wise precisa
- SSIM: Preserva struttura percettiva

### ✅ CombinedPerceptualLoss (L1 + SSIM + VGG Perceptual)

Loss avanzata con feature matching VGG16 per qualità percettiva.

**Formula:**

```python
loss = α * L1 + β * (1 - SSIM) + γ * Perceptual(VGG)
```

**Implementazione:**

- VGG16 pre-trained (ImageNet weights)
- Feature extraction: relu2_2, relu3_3 layers
- Smart γ=0 handling: usa CombinedLoss direttamente (no VGG overhead)

**Configurazione tipica:**

- α = 0.6 (L1)
- β = 0.25 (SSIM)
- γ = 0.15 (Perceptual) — 0 per disabilitare

**Vantaggi:**

- Migliore qualità visiva su texture complesse
- Riduce artifacts perceptually unpleasant

---

## 📊 Metriche di Valutazione

### Implementate

- **PSNR** (Peak Signal-to-Noise Ratio) — qualità pixel-wise (dB)
- **SSIM** (Structural Similarity Index) — similarità strutturale (0-1)
- **Sliding window inference** — valutazione su full-resolution images

### Valutazione Full-Resolution

- Patch size: 128×128 con overlap 32px
- Blending: weighted averaging nelle overlap regions
- Output: restored images salvate + metrics JSON

### Future Metrics (Planned)

- **LPIPS** (Learned Perceptual Image Patch Similarity)
- **FID** (Fréchet Inception Distance, per GAN)
- **Tempo di inferenza** e utilizzo memoria

---

## 🚀 Features del Training System

### ✅ Implementato

**Path Management Automatico:**

- `generate_degraded_dataset_auto()`: genera paths automatici basati su parametri
- Gaussian: `data/degraded/gaussian/sigma_{sigma}/`
- Dithering: `data/degraded/dithering/{type}/{bits}bit/`
- Existence checking: skip rigenerazione se dataset esiste

**Training Pipeline Avanzato:**

- **Mixed Precision (AMP)**: training più veloce con FP16/FP32
- **Warmup scheduling**: linear warmup + cosine annealing
- **Early stopping**: patience-based con best model tracking
- **Gradient clipping**: max_norm=1.0 per stabilità
- **Checkpointing**: best model + periodic saves

**Experiment Management:**

- Auto-naming: `{timestamp}_{custom_name}/`
- Config saving: JSON per reproducibility
- History tracking: loss curves + learning rate
- TensorBoard logging: metriche real-time

**Telegram Notifications:**

- Notifiche automatiche ogni N epochs
- Metrics summary (loss, PSNR, SSIM)
- Training progress tracking

**Data Augmentation:**

- Random crops (128×128 patches)
- Random horizontal/vertical flips
- Normalization [-1, 1]

### 🔧 Hyperparameters Tipici

**UNet / UNet Residual:**

```python
batch_size: 16
learning_rate: 1e-4
weight_decay: 1e-5
warmup_epochs: 5
scheduler: cosine
patience: 5
```

**Attention UNet:**

```python
batch_size: 16
learning_rate: 3e-5 - 5e-5  # Più basso per stabilità
weight_decay: 1e-6          # Ridotto per 13M+ params
warmup_epochs: 8            # Warmup più lungo
patience: 5-6
```

---

## 🚀 Setup e Installazione

### Requirements

- Python 3.8+
- CUDA 11.8+ (per training GPU)
- 8GB+ RAM
- 4GB+ VRAM (consigliato per batch_size=16)

### Installazione

```bash
# Clone repository
git clone https://github.com/GiuseppeBellamacina/Image-Enhancement.git
cd Image-Enhancement

# Crea virtual environment (opzionale ma consigliato)
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate  # Linux/Mac

# Installa dipendenze
pip install -r requirements.txt

# Oppure usa setup script (Windows + uv)
.\setup.ps1
```

### Framework e Librerie Principali

**Core:**

- `torch>=2.0.0` — PyTorch framework
- `torchvision>=0.15.0` — Pre-trained models e transforms
- `pytorch-msssim` — SSIM loss differenziabile

**Image Processing:**

- `opencv-python` — I/O immagini e processing
- `Pillow` — Image loading
- `scikit-image` — Metriche (PSNR, SSIM)

**Utilities:**

- `tqdm` — Progress bars
- `tensorboard` — Experiment logging
- `matplotlib`, `seaborn` — Visualizzazione
- `requests` — Dataset download

**Development:**

- `ruff` — Linting e formatting
- `jupyter` — Notebook experiments

---

## 📊 Dataset

### ✅ DIV2K (In uso)

[**DIV2K**](https://data.vision.ee.ethz.ch/cvl/DIV2K/) — High-quality image restoration dataset

**Caratteristiche:**

- 800 immagini training (2K resolution)
- 100 immagini validation (2K resolution)
- High quality, diverse content
- Download automatico tramite `download_div2k_dataset()`

**Storage:**

```
data/raw/
├── DIV2K_train_HR/  # 800 images
└── DIV2K_valid_HR/  # 100 images
```

---

## 👥 Team

**Membri del gruppo:**

- Giuseppe Bellamacina — Unet, UNet Residual, Attention UNet, Loss Functions, Training System, Evaluation
- Daniele Barbagallo — Pix2Pix GAN, Transformer-based models
- Salvatore Iurato — DnCNN
- Mattia Campanella — Denoising Autoencoder

---

## 📅 Timeline (bozza)

| Settimana | Milestone                               |
| --------- | --------------------------------------- |
| **1**     | Dataset, degradazioni, repository setup |
| **2**     | CNN + UNet + baseline classici          |
| **3**     | GAN / Transformer + loss percettive     |
| **4**     | Training completo + metriche            |
| **5**     | Ablation study + analisi                |
| **6**     | Relazione finale + presentazione        |

---

## 📘 Stato Attuale del Progetto

### ✅ Completato

- [x] Repository setup + structure
- [x] DIV2K dataset integration + auto-download
- [x] Path management system automatico
- [x] Gaussian noise degradation (parametrizzabile)
- [x] Quantization dithering degradation (bits + type)
- [x] Modelli: UNet, UNet Residual, Attention UNet
- [x] Loss functions: L1+SSIM, Perceptual Loss (VGG16)
- [x] Training pipeline completo (AMP, warmup, scheduler, early stopping)
- [x] Evaluation system (sliding window, PSNR/SSIM)
- [x] Experiment management (checkpointing, logging, TensorBoard)
- [x] Telegram notifications
- [x] Jupyter notebooks per testing

### 🔄 In Corso

- [ ] Training Attention UNet con perceptual loss
- [ ] Ablation study: loss functions comparison
- [ ] Ablation study: architecture comparison

### 🔜 Prossimi Steps

- [ ] Implementare degradazioni aggiuntive (blur, JPEG, low-light)
- [ ] Testare modelli aggiuntivi (DnCNN, opzionalmente GAN/Transformer)
- [ ] Ablation study completo
- [ ] Relazione finale + presentazione

---

## 📖 Usage

### 1. Download Dataset

Il dataset DIV2K viene scaricato automaticamente al primo training, oppure manualmente:

```python
from src.utils import download_div2k_dataset

download_div2k_dataset()
# Download in data/raw/DIV2K_train_HR e DIV2K_valid_HR
```

### 2. Genera Dataset Degradato

Sistema di path automatico basato su parametri:

```python
from src.degradations import generate_degraded_dataset_auto

# Gaussian Noise
train_deg, train_clean = generate_degraded_dataset_auto(
    dataset_split="DIV2K_train_HR",
    degradation_type="gaussian_noise",
    noise_sigma=100.0,  # Auto-path: gaussian/sigma_100/
    seed=42
)

# Quantization Dithering
train_deg, train_clean = generate_degraded_dataset_auto(
    dataset_split="DIV2K_train_HR",
    degradation_type="quantization_dithering",
    bits_per_channel=4,      # 4-bit quantization
    dithering_type="random", # Auto-path: dithering/random/4bit/
    seed=42
)
```

**Vantaggi:**

- Path generation automatica basata su parametri
- Existence checking: skip se dataset esiste già
- Consistent naming convention

### 3. Training

Usa i notebook in `notebooks/test_trainings/` per esempi completi.

**Quick Start - UNet su Gaussian Noise:**

```python
from src.models import UNet
from src.losses import CombinedLoss
from src.training import get_dataloaders, run_training

# Setup model
model = UNet(in_channels=3, out_channels=3, features=64, bilinear=True)

# Loss function
criterion = CombinedLoss(alpha=0.84, beta=0.16)

# Dataloaders
train_loader, val_loader = get_dataloaders(
    train_degraded_dir="data/degraded/gaussian/sigma_100/DIV2K_train_HR",
    train_clean_dir="data/raw/DIV2K_train_HR",
    val_degraded_dir="data/degraded/gaussian/sigma_100/DIV2K_valid_HR",
    val_clean_dir="data/raw/DIV2K_valid_HR",
    batch_size=16,
    patch_size=128,
    patches_per_image=20
)

# Training
history, best_info = run_training(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device="cuda",
    num_epochs=36,
    use_amp=True
)
```

**Con Perceptual Loss:**

```python
from src.losses import CombinedPerceptualLoss

criterion = CombinedPerceptualLoss(
    alpha=0.6,   # L1 weight
    beta=0.25,   # SSIM weight
    gamma=0.15,  # Perceptual weight (0 per disabilitare)
    vgg_layers=["relu2_2", "relu3_3"]
)
```

### 4. Evaluation

```python
from src.evaluation import ImageRestorationEvaluator

# Setup evaluator
evaluator = ImageRestorationEvaluator(
    model=model,
    device="cuda",
    patch_size=128,
    overlap=32  # Overlap per smooth blending
)

# Evaluate su validation set
results = evaluator.evaluate_dataset(
    degraded_dir="data/degraded/gaussian/sigma_100/DIV2K_valid_HR",
    clean_dir="data/raw/DIV2K_valid_HR",
    output_dir="experiments/unet/gaussian/restored_images",
    save_outputs=True
)

# Print summary
evaluator.print_summary(results)
# Output: Average PSNR, SSIM + per-image metrics
```

### 5. TensorBoard Monitoring

```bash
tensorboard --logdir experiments/
```

Visualizza:

- Training/validation loss curves
- Learning rate schedule
- PSNR/SSIM metrics
- Sample images

---

## 📚 Riferimenti

**Architetture:**

- **UNet**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **Attention UNet**: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas" (2018)
- **Residual Learning**: He et al., "Deep Residual Learning for Image Recognition" (2016)

**Loss Functions:**

- **SSIM**: Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (2004)
- **Perceptual Loss**: Johnson et al., "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (2016)
- **VGG Features**: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (2018)

**Image Restoration:**

- **DnCNN**: Zhang et al., "Beyond a Gaussian Denoiser" (2017)
- **Noise2Noise**: Lehtinen et al., "Noise2Noise: Learning Image Restoration without Clean Data" (2018)

**Datasets:**

- **DIV2K**: Agustsson & Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution" (2017)

---

## 👥 Author

**Giuseppe Bellamacina**

Progetto sviluppato per il corso di **Deep Learning** — A.A. 2025/2026

---

## 📎 License

MIT License

Copyright (c) 2025 Giuseppe Bellamacina

---

## 🙏 Acknowledgments

- PyTorch team per il framework
- DIV2K dataset creators
- pytorch-msssim library per SSIM differenziabile
- VS Code + Copilot per development support

---

**Note:** Progetto in sviluppo attivo. README aggiornato regolarmente con nuove features e risultati.
