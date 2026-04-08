import json
import re

nb_path = r'i:\Development 2.0\Deep_Learning\Progetto1\Image-Enhancement\notebooks\test_trainings\nafnet\gaussian\nafnet_gaussian.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = "".join(cell['source'])
        
        # 1. Update imports
        if 'from src.losses.combined_loss import CombinedLoss' in source:
            source = source.replace('from src.losses.combined_loss import CombinedLoss', 'from src.losses.combined_loss import get_criterion')
        
        # 2. Update config block
        if 'config = {' in source and '"model_name": "nafnet"' in source:
            source = """config = {
    # Training - Rifacciamo la Fase 1 da zero con DropPath e Charbonnier
    "resume_from_checkpoint": False,
    "resume_experiment": "latest",

    "batch_size": 32,
    "num_epochs": 200,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,

    # Data
    "patch_size": 128,
    "patches_per_image": 1,
    "num_workers": 0,

    # Model
    "model_name": "nafnet",
    "naf_width": 64,
    "naf_middle_blocks": 1,
    "naf_enc_blocks": (1, 1, 2, 8),
    "naf_dec_blocks": (1, 1, 1, 1),
    "naf_dw_expand": 2,
    "naf_ffn_expand": 2,
    "naf_drop_out_rate": 0.0,
    "naf_drop_path_rate": 0.05,     # <-- RISOLTO: ORA VIENE PASSATO AL MODELLO

    # Loss - Usiamo Charbonnier come nel paper originale di NAFNet
    "loss_type": "charbonnier",
    "loss_alpha": 1.0,
    "loss_beta": 0.0,              # Niente SSIM in Fase 1

    # Degradation
    "degradation_type": "gaussian_noise",
    "noise_sigma": 100.0,

    # Optimization
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    "gradient_clip": 1.0,

    # Early stopping
    "patience": 30,

    # Checkpoints
    "save_every": 10,
    "val_every": 1,

    # Mixed Precision
    "use_amp": True,
    "perf_log_every": 5,
    "perf_sync_cuda": False,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}"""
            
        # 3. Update model init
        if 'model = NAFNet(' in source:
            source = source.replace(
                '    drop_out_rate=config["naf_drop_out_rate"],\n).to(config["device"]) ##',
                '    drop_out_rate=config["naf_drop_out_rate"],\n    drop_path_rate=config.get("naf_drop_path_rate", 0.0),\n).to(config["device"])'
            )
            if 'criterion = CombinedLoss(alpha=config["loss_alpha"], beta=config["loss_beta"]).to(' in source:
                source = re.sub(
                    r'criterion = CombinedLoss\(alpha=config\["loss_alpha"\], beta=config\["loss_beta"\]\)\.to\(\s*config\["device"\]\s*\)',
                    'criterion = get_criterion(config).to(config["device"])',
                    source,
                    flags=re.MULTILINE
                )
                
        # 4. Comment out fine tuning block
        if 'load_pretrained_model(' in source and 'FINE-TUNING' in source:
            lines = source.split('\n')
            new_lines = ['# FINE-TUNING DISABILITATO: Alleniamo da zero la Fase 1']
            for line in lines[1:]:
                new_lines.append('# ' + line)
            source = '\n'.join(new_lines)

        # Convert back to list of lines with newlines
        lines = source.split('\n')
        cell['source'] = [line + '\n' if i < len(lines)-1 else line for i, line in enumerate(lines)]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook updating using json finished!')
