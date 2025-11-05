# Version Selection Report

## Selected Versions for Cloud Deployment

### Core Dependencies

| Package | Version | Rationale |
|---------|---------|-----------|
| **PyTorch** | 2.1.2 | Stable, widely tested, excellent CUDA support |
| **TorchVision** | 0.16.2 | Matches PyTorch 2.1.2 |
| **NumPy** | 1.24.3 | Stable, compatible with PyTorch 2.1+, avoids breaking changes |
| **Python** | 3.10 | Stable, widely supported on cloud platforms |

### CUDA Support

**Target CUDA Version:** 11.8 (primary) or 12.1 (alternative)

- **CUDA 11.8**: Most widely available on cloud platforms (AWS, GCP, TAMU HPRC)
- **CUDA 12.1**: Newer systems, better performance
- PyTorch 2.1.2 supports both versions

### Cloud Platform Compatibility

#### ✅ AWS EC2/ECS
- Supports CUDA 11.8 and 12.1
- Tested on: g4dn, p3, p4 instances
- Works with Deep Learning AMIs

#### ✅ Google Cloud Platform
- Supports CUDA 11.8 and 12.1
- Compatible with GPU-enabled Compute Engine instances
- Works with Google Cloud Deep Learning VM images

#### ✅ TAMU HPRC
- Standard GPU nodes typically have CUDA 11.8+
- Compatible with SLURM job scheduler
- Should work with existing HPRC GPU infrastructure

### Installation Options

#### Option 1: CUDA 11.8 (Recommended for maximum compatibility)
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

#### Option 2: CUDA 12.1 (For newer systems)
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

#### Option 3: CPU-only (For testing without GPU)
```bash
pip install torch==2.1.2 torchvision==0.16.2
```

### Version Selection Criteria

1. **Stability**: Chose 2.1.2 over 2.2+ or 2.7+ for proven stability
2. **Cloud Compatibility**: CUDA 11.8 is the most common version across platforms
3. **Ecosystem**: NumPy 1.24.3 is well-tested with PyTorch 2.1
4. **Support**: These versions have good documentation and community support

### Upgrade Notes

**From PyTorch 0.4.0 → 2.1.2:**
- Python 3.5 → 3.10 (required)
- NumPy 1.14.5 → 1.24.3 (significant upgrade)
- Ubuntu 16.04 → 22.04 (for Docker)

### Testing Recommendations

Before deploying to cloud, test:
1. ✅ Model loading from old checkpoints (with map_location)
2. ✅ CUDA availability and device detection
3. ✅ Data loading pipeline
4. ✅ Training loop execution
5. ✅ Evaluation scripts

### Alternative Versions (If Issues Arise)

**More Conservative:**
- PyTorch 2.0.1 (very stable, slightly older)
- NumPy 1.23.5 (if compatibility issues)

**More Recent:**
- PyTorch 2.2.0 (newer features, still stable)
- NumPy 1.26.0 (latest, may have minor compatibility considerations)

---

**Last Updated:** 2024
**Selected By:** AI Assistant for maximum cloud compatibility
