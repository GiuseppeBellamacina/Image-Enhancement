"""
Training loop for Pix2Pix GAN.

Implements alternating training of Generator and Discriminator with
adversarial loss + L1 reconstruction loss.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .training_utils import (
    cleanup_cuda_memory,
    handle_oom_error,
    create_progress_bar,
    apply_gradient_clipping_optimizer,
)


def train_epoch_pix2pix(
    generator,
    discriminator,
    train_loader,
    criterion_GAN,
    criterion_L1,
    optimizer_G,
    optimizer_D,
    device,
    epoch,
    lambda_L1=100.0,
    gradient_clip=1.0,
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
        epoch: Current epoch number (for progress display)
        lambda_L1: Weight for L1 loss (default: 100.0)
        gradient_clip: Maximum gradient norm for clipping (default: 1.0)
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
    
    pbar = create_progress_bar(train_loader, epoch, phase="Train", leave=False, position=1)
    
    for batch_idx, (degraded, clean) in enumerate(pbar):
        output_fake = None
        loss_D = None
        loss_G = None
        
        try:
            batch_size = degraded.size(0)
            degraded = degraded.to(device)
            clean = clean.to(device)
            
            # ====================================
            # Train Discriminator
            # ====================================
            optimizer_D.zero_grad()
            
            if use_amp:
                with autocast():
                    # Real pairs (degraded, clean)
                    pred_real = discriminator(degraded, clean)
                    # Labels for real and fake (match discriminator output size)
                    real_label = torch.ones_like(pred_real, device=device)
                    fake_label = torch.zeros_like(pred_real, device=device)
                    loss_D_real = criterion_GAN(pred_real, real_label)
                    
                    # Fake pairs (degraded, generated)
                    fake_images = generator(degraded)
                    pred_fake = discriminator(degraded, fake_images.detach())  # Detach to avoid backprop to G
                    loss_D_fake = criterion_GAN(pred_fake, fake_label)
                    
                    # Total discriminator loss
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                
                scaler_D.scale(loss_D).backward()
                
                # Gradient clipping for discriminator
                apply_gradient_clipping_optimizer(
                    optimizer_D,
                    discriminator.parameters(),
                    max_norm=gradient_clip,
                    scaler=scaler_D
                )
                
                scaler_D.step(optimizer_D)
                scaler_D.update()
            else:
                # Real pairs (degraded, clean)
                pred_real = discriminator(degraded, clean)
                # Labels for real and fake (match discriminator output size)
                real_label = torch.ones_like(pred_real, device=device)
                fake_label = torch.zeros_like(pred_real, device=device)
                loss_D_real = criterion_GAN(pred_real, real_label)
                
                # Fake pairs (degraded, generated)
                fake_images = generator(degraded)
                pred_fake = discriminator(degraded, fake_images.detach())  # Detach to avoid backprop to G
                loss_D_fake = criterion_GAN(pred_fake, fake_label)
                
                # Total discriminator loss
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                
                # Gradient clipping for discriminator
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=gradient_clip)
                
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
                
                # Gradient clipping for generator
                apply_gradient_clipping_optimizer(
                    optimizer_G,
                    generator.parameters(),
                    max_norm=gradient_clip,
                    scaler=scaler_G
                )
                
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
                
                # Gradient clipping for generator
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=gradient_clip)
                
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
        
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Check if it's an OOM error
            if "out of memory" in str(e).lower():
                pbar.close()
                handle_oom_error(
                    batch_idx,
                    len(train_loader),
                    device,
                    degraded,
                    clean,
                    output_fake,
                    loss_D,
                    loss_G,
                    is_training=True
                )
            else:
                # Re-raise non-OOM RuntimeErrors
                raise
    
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
    epoch,
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
        epoch: Current epoch number (for progress display)
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
    
    pbar = create_progress_bar(val_loader, epoch, phase="Val", leave=False, position=1)
    
    for batch_idx, (degraded, clean) in enumerate(pbar):
        fake_images = None
        loss_G = None
        loss_D = None
        
        try:
            batch_size = degraded.size(0)
            degraded = degraded.to(device)
            clean = clean.to(device)
            
            # Generator forward
            fake_images = generator(degraded)
            
            # Discriminator predictions
            pred_real = discriminator(degraded, clean)
            pred_fake = discriminator(degraded, fake_images)
            
            # Labels (match discriminator output size)
            real_label = torch.ones_like(pred_real, device=device)
            fake_label = torch.zeros_like(pred_fake, device=device)
            
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
        
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Check if it's an OOM error
            if "out of memory" in str(e).lower():
                pbar.close()
                handle_oom_error(
                    batch_idx,
                    len(val_loader),
                    device,
                    degraded,
                    clean,
                    fake_images,
                    loss_G,
                    loss_D,
                    is_training=False
                )
            else:
                # Re-raise non-OOM RuntimeErrors
                raise
    
    # Calculate averages
    n_batches = len(val_loader)
    metrics = {
        'loss_G': total_loss_G / n_batches,
        'loss_G_GAN': total_loss_G_GAN / n_batches,
        'loss_G_L1': total_loss_G_L1 / n_batches,
        'loss_D': total_loss_D / n_batches,
    }
    
    return metrics
