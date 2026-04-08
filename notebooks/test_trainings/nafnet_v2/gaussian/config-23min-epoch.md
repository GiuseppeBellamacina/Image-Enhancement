config = {
    # Resume Training
    "resume_from_checkpoint": False,
    "resume_experiment": "latest",
    
    # Training
    "batch_size": 8,              # ⬇ Ridotto: width=64 con enc (1,1,2,8) usa molta VRAM
    "num_epochs": 200,            # ⬆ Molto più lungo: NAFNet ha bisogno di convergere bene
    "learning_rate": 1e-3,        # ✅ Buono per AdamW con cosine schedule
    "weight_decay": 0.0,          # ⬇ Zero! NAFNet usa LayerNorm, non serve regolarizzazione extra
    
    # Data
    "patch_size": 256,            # ⬆ Patch più grandi = contesto spaziale maggiore per σ=100
    "patches_per_image": 4,       # ⬆ Più patch per epoca = training più efficace su DIV2K (solo 800 img)
    
    # IMPORTANT (Windows/Jupyter): keep 0 to avoid DataLoader worker freeze at first batch
    "num_workers": 0,
    
    # Model - Configurazione NAFSSR-B (Base) come da piano
    "model_name": "nafnet",
    "naf_width": 64,              # ✅ Bene, serve capacità per σ=100
    "naf_middle_blocks": 1,       # ✅ OK
    "naf_enc_blocks": (1, 1, 2, 8),  # ✅ Buona configurazione, abbastanza profonda
    "naf_dec_blocks": (1, 1, 1, 1),  # ✅ Decoder leggero come nel paper
    "naf_dw_expand": 2,           # ✅ Standard NAFNet
    "naf_ffn_expand": 2,          # ✅ Standard NAFNet
    "naf_drop_out_rate": 0.0,     # ✅ Niente dropout, c'è già stochastic depth
    "naf_drop_path_rate": 0.05,   # 🆕 STOCHASTIC DEPTH! Fondamentale su DIV2K piccolo
    
    # Loss - SOLO L1 come raccomandato dal piano!
    "loss_alpha": 1.0,            # ⬆ SOLO L1 nella fase iniziale
    "loss_beta": 0.0,             # ⬇ NIENTE SSIM inizialmente (aggiungere solo nel fine-tuning)
    
    # Degradation - Gaussian Noise
    "degradation_type": "gaussian_noise",
    "noise_sigma": 100.0,
    
    # Optimization
    "scheduler": "cosine",
    "warmup_epochs": 5,           # ⬆ Warmup! Fondamentale con LR alta per stabilizzare
    "min_lr": 1e-6,               # ✅ OK
    "gradient_clip": 1.0,         # ✅ OK, protegge da gradient explosion
    
    # Early stopping
    "patience": 30,               # ⬆ Più pazienza: cosine schedule può avere plateau lunghi
    
    # Checkpoints
    "save_every": 10,
    
    # Validate every epoch for immediate feedback
    "val_every": 1,               # ⬇ Ogni epoca! Così vedi subito se converge
    
    # Mixed Precision
    "use_amp": True,              # ✅ Essenziale con width=64
    
    # Performance logging
    "perf_log_every": 5,
    "perf_sync_cuda": False,
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}
