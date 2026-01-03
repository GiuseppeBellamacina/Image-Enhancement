"""
Training loop for Pix2Pix GAN.

Implements alternating training of Generator and Discriminator with
adversarial loss + L1 reconstruction loss.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def train_epoch_pix2pix(
    generator,
    discriminator,
    train_loader,
    criterion_GAN,
    criterion_L1,
    optimizer_G,
    optimizer_D,
    device,
    lambda_L1=100.0,
    use_amp=False,
    scaler_G=None,
    scaler_D=None,
):
    """
    Train Pix2Pix GAN for one epoch.
    
    Args:
        generator: Generator network (UNet-based)
        discriminator: Discriminator network (PatchGAN)
        train_loader: Training data loader
        criterion_GAN: Adversarial loss (BCELoss)
        criterion_L1: Reconstruction loss (L1Loss)
        optimizer_G: Generator optimizer
        optimizer_D: Discriminator optimizer
        device: Training device
        lambda_L1: Weight for L1 loss (default: 100.0)
        use_amp: Use automatic mixed precision
        scaler_G: Gradient scaler for generator
        scaler_D: Gradient scaler for discriminator
        
    Returns:
        Dictionary with average metrics for the epoch
    """
    generator.train()
    discriminator.train()
    
    # Metrics accumulators
    total_loss_G = 0.0
    total_loss_G_GAN = 0.0
    total_loss_G_L1 = 0.0
    total_loss_D = 0.0
    total_loss_D_real = 0.0
    total_loss_D_fake = 0.0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (degraded, clean) in enumerate(pbar):
        batch_size = degraded.size(0)
        degraded = degraded.to(device)
        clean = clean.to(device)
        
        # Labels for real and fake
        real_label = torch.ones(batch_size, 1, 30, 30, device=device)  # PatchGAN output size
        fake_label = torch.zeros(batch_size, 1, 30, 30, device=device)
        
        # ====================================
        # Train Discriminator
        # ====================================
        optimizer_D.zero_grad()
        
        if use_amp:
            with autocast():
                # Real pairs (degraded, clean)
                pred_real = discriminator(degraded, clean)
                loss_D_real = criterion_GAN(pred_real, real_label)
                
                # Fake pairs (degraded, generated)
                fake_images = generator(degraded)
                pred_fake = discriminator(degraded, fake_images.detach())  # Detach to avoid backprop to G
                loss_D_fake = criterion_GAN(pred_fake, fake_label)
                
                # Total discriminator loss
                loss_D = (loss_D_real + loss_D_fake) * 0.5
            
            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
        else:
            # Real pairs (degraded, clean)
            pred_real = discriminator(degraded, clean)
            loss_D_real = criterion_GAN(pred_real, real_label)
            
            # Fake pairs (degraded, generated)
            fake_images = generator(degraded)
            pred_fake = discriminator(degraded, fake_images.detach())  # Detach to avoid backprop to G
            loss_D_fake = criterion_GAN(pred_fake, fake_label)
            
            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
        
        # ====================================
        # Train Generator
        # ====================================
        optimizer_G.zero_grad()
        
        if use_amp:
            with autocast():
                # Generate fake images
                fake_images = generator(degraded)
                
                # Adversarial loss: fool discriminator
                pred_fake = discriminator(degraded, fake_images)
                loss_G_GAN = criterion_GAN(pred_fake, real_label)  # Want D to think it's real
                
                # L1 reconstruction loss
                loss_G_L1 = criterion_L1(fake_images, clean)
                
                # Total generator loss
                loss_G = loss_G_GAN + lambda_L1 * loss_G_L1
            
            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
        else:
            # Generate fake images
            fake_images = generator(degraded)
            
            # Adversarial loss: fool discriminator
            pred_fake = discriminator(degraded, fake_images)
            loss_G_GAN = criterion_GAN(pred_fake, real_label)  # Want D to think it's real
            
            # L1 reconstruction loss
            loss_G_L1 = criterion_L1(fake_images, clean)
            
            # Total generator loss
            loss_G = loss_G_GAN + lambda_L1 * loss_G_L1
            loss_G.backward()
            optimizer_G.step()
        
        # Accumulate metrics
        total_loss_G += loss_G.item()
        total_loss_G_GAN += loss_G_GAN.item()
        total_loss_G_L1 += loss_G_L1.item()
        total_loss_D += loss_D.item()
        total_loss_D_real += loss_D_real.item()
        total_loss_D_fake += loss_D_fake.item()
        
        # Update progress bar
        pbar.set_postfix({
            'G': f'{loss_G.item():.4f}',
            'D': f'{loss_D.item():.4f}',
            'L1': f'{loss_G_L1.item():.4f}'
        })
    
    # Calculate averages
    n_batches = len(train_loader)
    metrics = {
        'loss_G': total_loss_G / n_batches,
        'loss_G_GAN': total_loss_G_GAN / n_batches,
        'loss_G_L1': total_loss_G_L1 / n_batches,
        'loss_D': total_loss_D / n_batches,
        'loss_D_real': total_loss_D_real / n_batches,
        'loss_D_fake': total_loss_D_fake / n_batches,
    }
    
    return metrics


@torch.no_grad()
def validate_pix2pix(
    generator,
    discriminator,
    val_loader,
    criterion_GAN,
    criterion_L1,
    device,
    lambda_L1=100.0,
):
    """
    Validate Pix2Pix GAN.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        val_loader: Validation data loader
        criterion_GAN: Adversarial loss
        criterion_L1: Reconstruction loss
        device: Device
        lambda_L1: Weight for L1 loss
        
    Returns:
        Dictionary with average validation metrics
    """
    generator.eval()
    discriminator.eval()
    
    total_loss_G = 0.0
    total_loss_G_GAN = 0.0
    total_loss_G_L1 = 0.0
    total_loss_D = 0.0
    
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    
    for degraded, clean in pbar:
        batch_size = degraded.size(0)
        degraded = degraded.to(device)
        clean = clean.to(device)
        
        # Labels
        real_label = torch.ones(batch_size, 1, 30, 30, device=device)
        fake_label = torch.zeros(batch_size, 1, 30, 30, device=device)
        
        # Generator forward
        fake_images = generator(degraded)
        
        # Discriminator predictions
        pred_real = discriminator(degraded, clean)
        pred_fake = discriminator(degraded, fake_images)
        
        # Losses
        loss_D_real = criterion_GAN(pred_real, real_label)
        loss_D_fake = criterion_GAN(pred_fake, fake_label)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        loss_G_GAN = criterion_GAN(pred_fake, real_label)
        loss_G_L1 = criterion_L1(fake_images, clean)
        loss_G = loss_G_GAN + lambda_L1 * loss_G_L1
        
        # Accumulate
        total_loss_G += loss_G.item()
        total_loss_G_GAN += loss_G_GAN.item()
        total_loss_G_L1 += loss_G_L1.item()
        total_loss_D += loss_D.item()
        
        pbar.set_postfix({
            'G': f'{loss_G.item():.4f}',
            'D': f'{loss_D.item():.4f}'
        })
    
    # Calculate averages
    n_batches = len(val_loader)
    metrics = {
        'loss_G': total_loss_G / n_batches,
        'loss_G_GAN': total_loss_G_GAN / n_batches,
        'loss_G_L1': total_loss_G_L1 / n_batches,
        'loss_D': total_loss_D / n_batches,
    }
    
    return metrics
