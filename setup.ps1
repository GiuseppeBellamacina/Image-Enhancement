# Setup ambiente Image Enhancement
# Installa torch CUDA 13.0 + dipendenze da pyproject.toml

Write-Host "=== Setup Image Enhancement ===" -ForegroundColor Cyan

# Step 1: Verifica UV
Write-Host "`nVerifica UV..." -ForegroundColor Yellow

$uvInstalled = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvInstalled) {
    Write-Host "UV non installato. Installazione..." -ForegroundColor Yellow
    pip install uv
    Write-Host "✅ UV installato!" -ForegroundColor Green
} else {
    Write-Host "✅ UV installato" -ForegroundColor Green
}

# Step 2: Rimuovi .venv esistente se presente
if (Test-Path ".venv") {
    Write-Host "`n⚠️  .venv esistente trovato. Rimuovere? (y/n)" -ForegroundColor Yellow
    $remove = Read-Host
    if ($remove -eq "y") {
        Remove-Item .venv -Recurse -Force
        Write-Host "✅ .venv rimosso" -ForegroundColor Green
    }
}

# Step 3: Crea venv
Write-Host "`nCreazione ambiente virtuale..." -ForegroundColor Yellow
uv venv
Write-Host "✅ Ambiente creato" -ForegroundColor Green

# Step 4: Attiva ambiente
Write-Host "`nAttivazione ambiente..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Step 5: Installa PyTorch CUDA 13.0
Write-Host "`nInstallazione PyTorch CUDA 13.0..." -ForegroundColor Cyan
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
Write-Host "✅ PyTorch CUDA installato" -ForegroundColor Green

# Step 6: Sincronizza dipendenze da pyproject.toml (include dev)
Write-Host "`nSincronizzazione dipendenze da pyproject.toml con uv sync..." -ForegroundColor Cyan
uv sync
Write-Host "✅ Dipendenze sincronizzate (base + dev)" -ForegroundColor Green

# Step 7: Verifica installazione
Write-Host "`nVerifica installazione..." -ForegroundColor Yellow

$verifyScript = @"
import torch
import cv2

print('\n' + '='*60)
print('AMBIENTE IMAGE ENHANCEMENT')
print('='*60)

print(f'\n🔥 PyTorch:')
print(f'   Version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   Device: {torch.cuda.get_device_name(0)}')

print(f'\n📦 Librerie:')
try:
    import cv2
    print(f'   ✓ OpenCV: {cv2.__version__}')
except: pass

try:
    import lpips
    print(f'   ✓ LPIPS installato')
except: pass

try:
    import tensorboard
    print(f'   ✓ TensorBoard installato')
except: pass

print('\n' + '='*60)
"@

python -c $verifyScript

# Info finali
Write-Host "`n=== Setup Completato! ===" -ForegroundColor Cyan
Write-Host "✅ PyTorch CUDA 13.0 installato" -ForegroundColor Green
Write-Host "✅ Dipendenze installate da pyproject.toml" -ForegroundColor Green

Write-Host "`nComandi utili:" -ForegroundColor Yellow
Write-Host "  Attiva ambiente: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  Disattiva: deactivate" -ForegroundColor White
Write-Host "  Aggiungi libreria: uv pip install nome-libreria" -ForegroundColor White
