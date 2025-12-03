# SGAN - Modern PyTorch Version

This is the upgraded version of Social GAN with modern PyTorch 2.1.2, fully device-agnostic (works on Mac/CPU and cloud GPUs).

## Quick Start

### Local Testing (Mac - CPU)

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation script to test
python scripts/evaluate_model.py --model_path models/sgan-models
```

### Docker Build

```bash
# Build image
docker build -t sgan_new .
docker build --platform linux/amd64 -t sgan-env . # for mac

# run container
docker run -it --rm -v "$(pwd)":/app sgan-env

# to exit:
exit
```

### Cloud Deployment

The code is ready for:
- AWS EC2/ECS (GPU instances)
- Google Cloud Platform (GPU-enabled)
- TAMU HPRC (GPU nodes)

See `VERSION_INFO.md` for version details and `MIGRATION_SUMMARY.md` for migration notes.

## Key Features

- ✅ **Device-Agnostic**: Works on CPU (Mac) and GPU (cloud) automatically
- ✅ **Modern PyTorch**: Upgraded to PyTorch 2.1.2
- ✅ **Backward Compatible**: Old checkpoints can be loaded
- ✅ **No Hardcoded CUDA**: All device handling is dynamic

## Version Information

- PyTorch: 2.1.2
- NumPy: 1.24.3
- Python: 3.10+
- CUDA: 11.8 or 12.1 (optional, for GPU)

See `VERSION_INFO.md` for complete details.


New:
```
# Load latest checkpoint from folder "eth"
python scripts/analyze_checkpoint.py --folder eth

# Load best checkpoint and generate plots
python scripts/analyze_checkpoint.py --folder eth --type best --plot

# Use full path
python scripts/analyze_checkpoint.py --folder checkpoints/eth --plot
```