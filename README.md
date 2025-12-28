# 📸 Image Enhancement with Deep Learning

**Corruption → Restoration → Evaluation**

Questo progetto di gruppo ha l'obiettivo di studiare e confrontare diversi metodi di **Image Enhancement** attraverso reti neurali e tecniche di degradazione controllata.

**L'idea centrale:**  
👉 Corrompere immagini in diversi modi  
👉 Ricostruirle usando modelli di deep learning  
👉 Confrontare i risultati con metriche quantitative e qualitative

---

## 🔍 Obiettivi del progetto

- [x] Implementare diversi **tipi di corruzione** delle immagini (rumore, blur, JPEG, low-light, ecc.)
- [x] Addestrare più **modelli di restauro** e enhancement: UNet, DnCNN, Autoencoder, GAN, Transformer
- [x] Testare varie **loss functions** (L1, L2, SSIM, Perceptual Loss)
- [x] Valutare la qualità ricostruita con metriche come **PSNR, SSIM, LPIPS**, e confronto visivo
- [x] Svolgere **ablation study** per comprendere cosa migliora o peggiora le performance
- [x] Redigere una **relazione finale** e presentazione del progetto

---

## 📁 Struttura della Repository

```
Image-Enhancement/
 ├── data/
 │    ├── raw/              # dataset originale (DIV2K, BSD500, ecc.)
 │    ├── degraded/         # immagini corrotte generate
 │    └── processed/        # patch / split pronti per training
 ├── src/
 │    ├── degradations/     # script per corruzioni immagini
 │    ├── models/           # architetture (UNet, DnCNN, GAN…)
 │    ├── losses/           # funzioni di loss custom
 │    ├── training/         # training loop + dataloader
 │    ├── evaluation/       # metriche e script di valutazione
 │    └── utils/            # funzioni ausiliarie
 ├── experiments/
 │    ├── configs/          # file YAML per ogni esperimento
 │    └── results/          # metriche .json, grafici, immagini
 ├── notebooks/
 │    └── analysis.ipynb    # analisi finale e ablation study
 ├── requirements.txt       # dipendenze Python
 └── README.md
```

---

## 🧪 Tipi di degradazione previsti

Saranno implementati diversi metodi di corruzione parametrizzabili:

- **Gaussian noise** (σ variabile: 5, 15, 25, 50)
- **Poisson noise**
- **Salt & Pepper** (densità 1%, 5%, 10%)
- **Gaussian blur** / **Motion blur** (kernel size e angoli variabili)
- **JPEG compression** (quality: 30, 50, 70, 90)
- **Low-light simulation** (scaling e gamma)
- **Haze/Fog** (atmospheric scattering)
- **Occlusioni casuali** (block dropout)
- **Quantizzazione + Dithering** (color quantization con diversi livelli di bit depth + dithering randomico)
  - Quantizzazione a 8, 6, 4, 2 bit per canale
  - Dithering: random, Floyd-Steinberg, Bayer pattern
  - Training: immagini pesantemente quantizzate/dithered → originali o versioni con dithering sofisticato
- **Combinazioni** (es. blur + noise, JPEG + salt-and-pepper)

---

## 🤖 Modelli di Image Enhancement previsti

### Baseline

- Filtri classici (median, gaussian, bilateral)
- BM3D (opzionale)

### CNN-based

- **Denoising Autoencoder**
- **UNet** (standard)
- **Residual UNet** (con skip connections)
- **DnCNN** (Denoising CNN)
- **Attention UNet** (opzionale)

### GAN-based

- **Pix2Pix** (conditional GAN)
- **SRGAN** / **ESRGAN** (opzionale)

### Transformer-based

- **SwinIR** (se tempo/GPU permettono)
- **IPT** (Image Processing Transformer, opzionale)

### Advanced (opzionale)

- **Noise2Noise** (self-supervised)
- **Diffusion models** per denoising

---

## 🎯 Metriche di valutazione

- **PSNR** (Peak Signal-to-Noise Ratio) — qualità pixel-wise
- **SSIM** (Structural Similarity Index) — similarità strutturale
- **LPIPS** (Learned Perceptual Image Patch Similarity) — similarità percettiva
- **FID** (Fréchet Inception Distance, per GAN)
- **Tempo di inferenza** e **memoria**
- Valutazioni **qualitative** con confronti visivi
- (Opzionale) **User study** interno al gruppo

---

## 🧩 Ablation Study previsto

- Confronto tra **loss functions** (L1 vs L2 vs L1+VGG vs SSIM)
- Effetto della **severità della corruzione** (curve PSNR vs livello)
- Differenze tra **architetture** (UNet vs DnCNN vs GAN)
- Training **single-degradation** vs **multi-degradation**
- Impatto della **dimensione delle patch** (128×128 vs 256×256)
- **Profondità della rete** / numero di parametri
- Training con **dati sintetici** vs **dati reali** (SIDD)

---

## 🚀 Setup e Requirements

### Installazione

```bash
# Clona la repository
git clone https://github.com/GiuseppeBellamacina/Image-Enhancement.git
cd Image-Enhancement

# Installa le dipendenze
pip install -r requirements.txt
```

### Framework e Librerie

- **Framework principale:** PyTorch
- **Librerie utili:**
  - `albumentations` — data augmentation
  - `timm` — backbone pretrainati
  - `lpips` — metriche percettive
  - `scikit-image` — processing e metriche
  - `opencv-python` — I/O immagini
  - `matplotlib`, `seaborn` — visualizzazione
  - `tensorboard` / `wandb` — logging esperimenti
  - `tqdm` — progress bar
  - `PyYAML` — config management

---

## 📊 Dataset utilizzati

- [**DIV2K**](https://data.vision.ee.ethz.ch/cvl/DIV2K/) — alta qualità per super-resolution/restoration
- **BSD500** — immagini naturali classiche
- **ImageNet** (subset) — ampia variabilità
- **SIDD** — real camera noisy images
- **GoPro** — motion blur / deblurring
- **LOL** — low-light enhancement (opzionale)

_I dataset saranno scaricati e posizionati in `data/raw/`_

---

## 👥 Team

**Membri del gruppo:**

- Giuseppe Bellamacina — _[ruolo da definire]_
- _[Altri membri]_

**Divisione iniziale dei compiti:**

- **Data & degradazioni** → [nome]
- **Modelli CNN/UNet** → [nome]
- **GAN/Transformer** → [nome]
- **Valutazione & relazione** → [nome]

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

## 📘 Stato attuale del progetto

- [x] Creazione repository
- [ ] Preparazione dataset
- [ ] Script degradazioni
- [ ] Modelli base (UNet, DnCNN)
- [ ] Training pipeline
- [ ] Metriche (PSNR, SSIM, LPIPS)
- [ ] Ablation study
- [ ] Relazione finale

---

## 📖 Usage

_(Da completare quando il codice sarà pronto)_

### Generare dataset degradato

```bash
python src/degradations/generate_degraded_dataset.py \
    --input data/raw/DIV2K \
    --output data/degraded \
    --corruption gaussian_noise \
    --severity 25
```

### Training di un modello

```bash
python src/training/train.py \
    --config experiments/configs/unet_gaussian_l1.yaml
```

### Valutazione

```bash
python src/evaluation/evaluate.py \
    --model experiments/results/unet_gaussian/best.pth \
    --test-dir data/processed/test
```

---

## 📚 Riferimenti e Paper

- **DnCNN**: Zhang et al., "Beyond a Gaussian Denoiser" (2017)
- **UNet**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **Pix2Pix**: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (2017)
- **SwinIR**: Liang et al., "SwinIR: Image Restoration Using Swin Transformer" (2021)
- **LPIPS**: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (2018)
- **Noise2Noise**: Lehtinen et al., "Noise2Noise: Learning Image Restoration without Clean Data" (2018)

---

## 📎 License

MIT License

Copyright (c) 2025 Giuseppe Bellamacina

---

## 🙏 Acknowledgments

Progetto sviluppato per il corso di **Deep Learning** — A.A. 2025/2026

---

**Note:** Questo progetto è in fase di sviluppo attivo. La documentazione verrà aggiornata man mano che implementiamo le varie componenti.
Image Enhancement project with PyTorch
