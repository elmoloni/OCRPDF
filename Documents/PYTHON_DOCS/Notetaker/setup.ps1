# One-off install of the Notetaker environment (~4 GB: PyTorch with CUDA, Whisper, pyannote).
# The venv lives outside OneDrive in %USERPROFILE%\.notetaker\venv so OneDrive does not sync it.
# Run:  powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
$root = Join-Path $env:USERPROFILE ".notetaker"
$venv = Join-Path $root "venv"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Creating venv in $venv (Python 3.12) ..."
    New-Item -ItemType Directory -Force $root | Out-Null
    py -3.12 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 not found (py -3.12). Install it from python.org first." }
}

& $py -m pip install --upgrade pip
# CUDA build of torch first (PyPI only has CPU builds for Windows); it also runs on machines without a GPU
& $py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
# faster-whisper's CTranslate2 is built against CUDA 12 / cuDNN 9, so it gets its own runtime DLLs
& $py -m pip install pyannote.audio faster-whisper PyAudioWPatch numpy nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

& $py -c "import torch, faster_whisper, pyannote.audio, pyaudiowpatch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '| pyannote', pyannote.audio.__version__)"
Write-Host ""
Write-Host "Done. Start the app with Notetaker.bat."
Write-Host "For speaker separation, log in to Hugging Face once:  $venv\Scripts\hf.exe auth login   (see README)"
