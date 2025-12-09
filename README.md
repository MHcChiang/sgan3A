# SGAN3A - Social GAN with AgentFormer Architecture

This repository contains the implementation of SGAN3A, a trajectory prediction model that combines Social GAN with AgentFormer architecture for pedestrian trajectory forecasting.

## Project Structure

```
sgan3A/
├── scripts/
│   ├── train_sgan3A.py      # Training script
│   ├── test_sgan3A.py        # Testing/evaluation script
│   └── analyze_checkpoint.py # Checkpoint analysis utility
├── model/                    # Model implementations
├── configs/                  # Configuration files
├── checkpoints/              # Saved model checkpoints
└── datasets/                 # Dataset files
```

## Scripts

### 1. `train_sgan3A.py`

Training script for the SGAN3A model. This script handles the complete training pipeline including data loading, model initialization, GAN training with optional warmup phase, checkpointing, and validation.

#### Key Features:
- GAN-based training with Generator and Discriminator
- Optional CVAE mode with KL divergence loss
- Warmup phase for generator-only training
- Learning rate schedulers (step, exponential, plateau, cosine)
- Data augmentation with random rotation
- Smart batching with agent-level limits
- Automatic checkpoint saving (latest and best models)

#### Usage:

**Basic training with config file:**
```bash
python scripts/train_sgan3A.py --cfg configs/ETH_sgan3A.yaml
```

**Training with command-line overrides:**
```bash
python scripts/train_sgan3A.py \
    --cfg configs/ETH_sgan3A.yaml \
    --batch_size 8 \
    --num_epochs 100 \
    --g_learning_rate 1e-4 \
    --d_learning_rate 1e-4 \
    --augment 1 \
    --use_gpu 1 \
    --gpu_num "0"
```

**Resume from checkpoint:**
```bash
python scripts/train_sgan3A.py \
    --cfg configs/ETH_sgan3A.yaml \
    --restore_from_checkpoint 1 \
    --checkpoint_start_from checkpoints/your_model/your_model_latest.pt
```

**Resume from warmup model:**
```bash
python scripts/train_sgan3A.py \
    --cfg configs/ETH_sgan3A.yaml \
    --resume_warmup_from warmed.pt
```

#### Key Parameters:
- `--cfg`: Path to YAML configuration file
- `--dataset`: Dataset name (default: 'eth')
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--augment`: Enable data augmentation (1) or disable (0)
- `--warmup_epochs`: Number of epochs for generator-only warmup
- `--k`: Number of samples for Variety Loss (Best-of-K)
- `--scheduler_type`: Learning rate scheduler type ('none', 'step', 'exponential', 'plateau', 'cosine')
- `--use_gpu`: Use GPU (1) or CPU (0)
- `--output_dir`: Directory to save checkpoints and logs

#### Output:
- Training logs saved to `{output_dir}/log.txt`
- Configuration saved to `{output_dir}/config_saved.yaml`
- Checkpoints saved as `{output_dir}/{checkpoint_name}_latest.pt` and `{output_dir}/{checkpoint_name}_best.pt`

---

### 2. `analyze_checkpoint.py`

Utility script to analyze and visualize training checkpoints. Extracts training metrics, losses, and generates diagnostic plots.

#### Key Features:
- Loads and validates checkpoint files
- Displays training progress summary
- Generates training curve visualizations (ADE/FDE, L2 loss, adversarial losses)
- Exports metrics to JSON format
- Supports both 'latest' and 'best' checkpoints

#### Usage:

**Analyze latest checkpoint:**
```bash
python scripts/analyze_checkpoint.py --folder checkpoints/1207_Aug_b4_WR1_K10
```

**Analyze best checkpoint:**
```bash
python scripts/analyze_checkpoint.py \
    --folder checkpoints/1207_Aug_b4_WR1_K10 \
    --type best
```

**Analyze with custom smoothing:**
```bash
python scripts/analyze_checkpoint.py \
    --folder checkpoints/1207_Aug_b4_WR1_K10 \
    --smooth-window 10
```

**Skip plotting:**
```bash
python scripts/analyze_checkpoint.py \
    --folder checkpoints/1207_Aug_b4_WR1_K10 \
    --no-plot
```

**Export metrics to JSON:**
```bash
python scripts/analyze_checkpoint.py \
    --folder checkpoints/1207_Aug_b4_WR1_K10 \
    --export-json metrics.json
```

#### Key Parameters:
- `--folder`: Folder name containing checkpoints (can be folder name or full path)
- `--type`: Checkpoint type to load ('latest' or 'best', default: 'latest')
- `--no-plot`: Skip generating training curve plots
- `--output-dir`: Directory to save plots (default: same as checkpoint folder)
- `--export-json`: Export metrics to JSON file
- `--smooth-window`: Window size for smoothing curves (default: 1)

#### Output:
- Console summary of checkpoint contents
- Training curves plot saved to `{checkpoint_folder}/training_curves.png` (unless `--no-plot` is specified)
- Optional JSON export with metrics

#### Generated Plots:
1. **Validation Accuracy**: ADE and FDE metrics over epochs
2. **G L2 Loss**: Training vs Validation L2 loss comparison
3. **Adversarial Losses**: Smoothed Generator adversarial loss and Discriminator loss

---

### 3. `test_sgan3A.py`

Testing and evaluation script for trained models. Evaluates model performance on test/validation sets and optionally generates trajectory visualizations.

#### Key Features:
- Loads trained model from checkpoint
- Evaluates using Best-of-K strategy
- Computes ADE (Average Displacement Error) and FDE (Final Displacement Error)
- Generates trajectory visualizations
- Supports CPU-only evaluation

#### Usage:

**Evaluate model (using best checkpoint):**
```bash
python scripts/test_sgan3A.py \
    --model_path checkpoints/1207_Aug_b4_WR1_K10 \
    --sample_k 20
```

**Evaluate using latest checkpoint:**
```bash
python scripts/test_sgan3A.py \
    --model_path checkpoints/1207_Aug_b4_WR1_K10 \
    --latest \
    --sample_k 20
```

**Generate trajectory visualizations:**
```bash
python scripts/test_sgan3A.py \
    --model_path checkpoints/1207_Aug_b4_WR1_K10 \
    --draw \
    --sample_k 20
```

**Evaluate with limited samples:**
```bash
python scripts/test_sgan3A.py \
    --model_path checkpoints/1207_Aug_b4_WR1_K10 \
    --sample_k 20 \
    --num_samples_check 1000
```

#### Key Parameters:
- `--model_path`: Path to model folder containing checkpoint and config (required)
- `--latest`: Use latest checkpoint instead of best checkpoint
- `--draw`: Generate trajectory visualization plots
- `--dataset`: Dataset name (default: 'eth')
- `--sample_k`: Number of samples for Best-of-K evaluation (default: 20)
- `--num_samples_check`: Limit number of samples to evaluate (None = all)
- `--shuffle`: Shuffle test data
- `--use_gpu`: Use GPU (0=CPU, 1=GPU, default: 0 for testing)

#### Output:
- Evaluation results logged to `{model_path}/log_test.txt`
- Trajectory visualization saved to `{model_path}/trajectory_viz.png` (if `--draw` is specified)
- Console output with ADE, FDE, and ADE+FDE metrics

#### Notes:
- The script automatically loads configuration from `config_saved.yaml` in the model folder
- Model architecture parameters are loaded from the saved config
- Evaluation uses CPU by default to avoid GPU memory issues during testing

---

## Configuration Files

Configuration files are stored in `configs/` directory as YAML files. Example: `configs/ETH_sgan3A.yaml`

Key configuration sections:
- **Data Parameters**: Dataset paths, frame settings, augmentation
- **Optimization**: Batch size, learning rates, training steps, warmup
- **Learning Rate Scheduler**: Scheduler type and parameters
- **Model Architecture**: Transformer dimensions, layers, latent space
- **CVAE Control**: CVAE mode and loss weights
- **Output**: Checkpoint directory and logging settings

---

## Checkpoint Structure

Checkpoints are saved as PyTorch `.pt` files containing:
- Model state dictionaries (generator and discriminator)
- Optimizer states
- Scheduler states
- Training metrics (losses, validation metrics)
- Training progress counters (epoch, iteration)
- Best model metrics (best ADE, best ADE+FDE)
- Training arguments/configuration

---

## Example Workflow

1. **Train a model:**
   ```bash
   python scripts/train_sgan3A.py --cfg configs/ETH_sgan3A.yaml
   ```

2. **Analyze training progress:**
   ```bash
   python scripts/analyze_checkpoint.py --folder checkpoints/your_model_name
   ```

3. **Evaluate the trained model:**
   ```bash
   python scripts/test_sgan3A.py --model_path checkpoints/your_model_name --sample_k 20
   ```

4. **Generate trajectory visualizations:**
   ```bash
   python scripts/test_sgan3A.py --model_path checkpoints/your_model_name --draw --sample_k 20
   ```
