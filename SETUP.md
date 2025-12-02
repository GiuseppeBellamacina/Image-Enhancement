# Image Enhancement Project - Setup Guide

Questa guida ti aiuterà a configurare il progetto da zero.

## 1. Setup Iniziale

### Clona il repository

```bash
git clone https://github.com/GiuseppeBellamacina/Image-Enhancement.git
cd Image-Enhancement
```

### Crea un ambiente virtuale Python

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Installa le dipendenze

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Download dei Dataset

### DIV2K (consigliato per iniziare)

```bash
# Scarica da: https://data.vision.ee.ethz.ch/cvl/DIV2K/
# Oppure usa script automatico:
python scripts/download_dataset.py --dataset DIV2K --output data/raw/DIV2K
```

### BSD500

```bash
# Download: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
```

## 3. Generazione Dataset Degradato

```bash
# Esempio: Gaussian noise
python src/degradations/generate_degraded_dataset.py \
    --input data/raw/DIV2K \
    --output data/degraded/gaussian \
    --corruption gaussian_noise \
    --sigma 25

# Esempio: Motion blur
python src/degradations/generate_degraded_dataset.py \
    --input data/raw/DIV2K \
    --output data/degraded/motion_blur \
    --corruption motion_blur \
    --kernel-size 15
```

## 4. Preprocessing (Patch Creation)

```bash
python src/training/make_patches.py \
    --clean-dir data/raw/DIV2K/train \
    --degraded-dir data/degraded/gaussian/train \
    --output data/processed/gaussian_patches \
    --patch-size 128 \
    --stride 64
```

## 5. Training

```bash
# UNet con Gaussian noise
python src/training/train.py \
    --config experiments/configs/unet_gaussian_l1.yaml

# DnCNN con Gaussian noise
python src/training/train.py \
    --config experiments/configs/dncnn_gaussian_l2.yaml

# Pix2Pix per motion blur
python src/training/train.py \
    --config experiments/configs/pix2pix_motion_blur.yaml
```

## 6. Valutazione

```bash
python src/evaluation/evaluate.py \
    --model experiments/results/unet_gaussian_l1/best_model.pth \
    --test-dir data/processed/test \
    --output experiments/results/unet_gaussian_l1/evaluation
```

## 7. Visualizzazione Risultati

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Checklist Progetto

### Week 1: Setup & Data

- [ ] Setup repository e ambiente Python
- [ ] Download dataset DIV2K
- [ ] Implementare script degradazioni (gaussian, blur, jpeg)
- [ ] Generare dataset degradati
- [ ] Creare patch per training

### Week 2: Baseline Models

- [ ] Implementare UNet
- [ ] Implementare DnCNN
- [ ] Implementare Autoencoder
- [ ] Training su Gaussian noise
- [ ] Calcolare metriche baseline

### Week 3: Advanced & GAN

- [ ] Implementare Pix2Pix
- [ ] Implementare Perceptual Loss
- [ ] Training GAN su blur
- [ ] Confronto CNN vs GAN

### Week 4: Ablation & Multiple Corruptions

- [ ] Training su motion blur, JPEG, salt-and-pepper
- [ ] Test combinazioni di loss (L1, L2, L1+VGG)
- [ ] Ablation study (depth, patch size, ecc.)
- [ ] Raccolta risultati

### Week 5: Evaluation & Analysis

- [ ] Calcolo metriche complete (PSNR, SSIM, LPIPS)
- [ ] Generazione grafici comparativi
- [ ] Selezione best/worst cases
- [ ] User study (opzionale)

### Week 6: Report & Presentation

- [ ] Stesura relazione
- [ ] Preparazione slide
- [ ] Revisione finale
- [ ] Presentazione

## Tips Pratici

### Debugging

```bash
# Test rapido su subset
python src/training/train.py --config experiments/configs/unet_gaussian_l1.yaml --debug --epochs 5

# Monitora training
tensorboard --logdir experiments/results
```

### Memory Management

- Usa `batch_size` più piccolo se vai in OOM (Out of Memory)
- Abilita `mixed_precision: true` nei config
- Riduci `patch_size` a 128x128 invece di 256x256

### Organizzazione Esperimenti

Ogni esperimento deve avere:

- Config YAML in `experiments/configs/`
- Checkpoint salvati in `experiments/results/<experiment_name>/checkpoints/`
- Log tensorboard in `experiments/results/<experiment_name>/logs/`
- Metriche JSON in `experiments/results/<experiment_name>/metrics.json`

## Troubleshooting

### CUDA out of memory

```yaml
# Nel config .yaml:
data:
  batch_size: 8 # Riduci da 16 a 8
  patch_size: 128 # Riduci da 256

training:
  mixed_precision: true # Abilita
```

### Import errors

```bash
# Assicurati di essere nell'ambiente virtuale
pip install -r requirements.txt --upgrade
```

### Dataset non trovato

```bash
# Verifica struttura:
ls data/raw/DIV2K/
# Dovrebbe contenere: train/, valid/
```

## Risorse Utili

- **Paper DnCNN**: https://arxiv.org/abs/1608.03981
- **Paper Pix2Pix**: https://arxiv.org/abs/1611.07004
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **LPIPS Metric**: https://github.com/richzhang/PerceptualSimilarity

## Contatti

Per domande o problemi, apri una issue su GitHub o contatta il team.
