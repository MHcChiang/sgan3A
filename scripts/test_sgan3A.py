import argparse
import os
import sys
import yaml
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append(os.getcwd())
from model.data.dataloader import data_generator, data_loader
from model.sgan3A import AgentFormerGenerator
from utils.logger import Logger

# Import helper functions from train_sgan3A.py
from scripts.train_sgan3A import (
    get_device,
    prepare_batch,
    collate_scenes,
    fetch_and_collate_batch,
    check_accuracy
)

# ==========================================
# 2. Evaluation Loop (Using check_accuracy from train_sgan3A.py)
# ==========================================
def evaluate(args, loader, generator):
    """
    Evaluate model using check_accuracy from train_sgan3A.py.
    """
    logger.info(f"Starting Evaluation (Best-of-{args.sample_k})...")
    
    # Use check_accuracy from train_sgan3A.py
    # Set num_samples_check if not already set (for limit parameter)
    if not hasattr(args, 'num_samples_check') or args.num_samples_check is None:
        args.num_samples_check = 500  # Default limit
    
    limit = args.num_samples_check > 0
    metrics = check_accuracy(args, loader, generator, limit=limit, k=args.sample_k, augment=False)
    
    return metrics['ade'], metrics['fde']


def draw_trajectory(args, loader, generator):
    """
    Visualizes 10 sequences with "Per-Agent Best" logic.
    - Past: Blue
    - GT Future: Green
    - All K Predictions: Light Red (Thin) [optional, controlled by draw_k]
    - Best Prediction (Per Agent): Dark Red (Bold)
    """
    logger.info("Drawing trajectories...")
    generator.eval()
    if args.shuffle:
        loader.shuffle()
    if hasattr(loader, 'reset'): loader.reset()
    else: loader.index = 0
    
    viz_samples = []
    device = torch.device('cpu')  # Force CPU for testing
    K = args.sample_k
    
    # --- 1. Collect 10 Samples ---
    with torch.no_grad():
        while len(viz_samples) < 10 and not loader.is_epoch_end():
            # Use fetch_and_collate_batch for consistency
            conn_dist = getattr(args, 'conn_dist', 100000.0)
            raw_batch = fetch_and_collate_batch(loader, batch_size=1, augment=False, conn_dist=conn_dist)
            if raw_batch is None: continue
            
            batch = prepare_batch(raw_batch, device)
            pred_real_abs = batch['fut_motion'] # [Time, Agents, 2]
            
            # Generate K samples
            batch_samples = []
            for _ in range(K):
                pred_fake, _ = generator(batch) 
                pred_fake = pred_fake.permute(1, 0, 2) # [Time, Agents, 2]
                batch_samples.append(pred_fake.unsqueeze(0))
            
            # Stack: [K, Time, Agents, 2]
            all_preds = torch.cat(batch_samples, dim=0) 
            
            # Find Best Prediction PER AGENT (min ADE)
            dist = torch.norm(all_preds - pred_real_abs.unsqueeze(0), dim=-1) # [K, Time, Agents]
            ade_per_agent = dist.mean(dim=1) # [K, Agents]
            best_indices = torch.argmin(ade_per_agent, dim=0) # [Agents]
            
            # Store data (CPU numpy)
            viz_samples.append({
                'past': batch['pre_motion'].cpu().numpy(),
                'gt': pred_real_abs.cpu().numpy(),
                'all_preds': all_preds.cpu().numpy(),
                'best_indices': best_indices.cpu().numpy()
            })

    # --- 2. Plotting ---
    if len(viz_samples) == 0:
        logger.info("No samples found to visualize.")
        return

    # Create subplot grid (2 rows, 5 cols)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    for i, sample in enumerate(viz_samples):
        if i >= 10: break
        ax = axes[i]
        
        past = sample['past']           # [Time, N, 2]
        gt = sample['gt']               # [Time, N, 2]
        all_preds = sample['all_preds'] # [K, Time, N, 2]
        best_indices = sample['best_indices'] # [N]
        
        num_agents = past.shape[1]
        
        for n in range(num_agents):
            # Capture the last observed point for gap filling
            last_obs = past[-1, n, :] # [2]
            
            # --- A. Plot Past (Blue) ---
            ax.plot(past[:, n, 0], past[:, n, 1], 'b-', linewidth=2, alpha=0.7, label='Past' if n==0 else "")
            
            # FIX 3: Mark Starting Point instead of End
            ax.scatter(past[0, n, 0], past[0, n, 1], c='b', s=20, marker='o') 

            # --- B. Plot GT Future (Green) ---
            # FIX 2: Fill Gap (Prepend last observation)
            gt_plot = np.vstack([last_obs, gt[:, n, :]])
            
            ax.plot(gt_plot[:, 0], gt_plot[:, 1], 'g-', linewidth=2, alpha=0.5, label='GT' if n==0 else "")
            ax.scatter(gt[-1, n, 0], gt[-1, n, 1], c='g', s=30, marker='*') # End marker

            # --- C. Plot All K Predictions (Light Red) [Optional] ---
            if args.draw_k:
                for k in range(K):
                    pred_k = all_preds[k, :, n, :] # [Time, 2]
                    pred_plot_k = np.vstack([last_obs, pred_k])
                    ax.plot(pred_plot_k[:, 0], pred_plot_k[:, 1], color='lightcoral', linewidth=0.5, alpha=0.3, label='All Preds' if (n==0 and k==0) else "")

            # --- D. Plot Best Prediction (Dark Red, Bold) ---
            best_k = best_indices[n]
            best_pred = all_preds[best_k, :, n, :] # [Time, 2]
            
            # FIX 2: Fill Gap
            pred_plot = np.vstack([last_obs, best_pred])
            
            ax.plot(pred_plot[:, 0], pred_plot[:, 1], 'r--', linewidth=2, color='darkred', label='Best Pred' if n==0 else "")

        ax.set_title(f"Scene {i+1} (N={num_agents})")
        ax.axis('equal')
        if i == 0: ax.legend()

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, 'trajectory_viz.png')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Trajectory visualization saved to: {save_path}")


def main(args):
    global logger
    
    # Setup Output
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    logger = Logger(os.path.join(output_dir, 'log_test.txt'))
    
    # Force CPU usage for testing
    device = torch.device('cpu')
    logger.info(f'Using device: {device} (forced CPU for testing)')

    # --- Load Model Config ---
    # Inject missing dictionaries needed by AgentFormerGenerator
    args.context_encoder = {
        'nlayer': args.enc_layers,
        'input_type': [args.input_type]
    }
    args.future_decoder = {
        'nlayer': args.dec_layers,
        'out_mlp_dim': [512, 256],
        'input_type': [args.input_type]
    }
    args.future_encoder = {
        'nlayer': args.enc_layers,
        'out_mlp_dim': [512, 256],
        'input_type': [args.input_type]
    }

    # Initialize Model
    generator = AgentFormerGenerator(args).to(device)
    
    # Load Checkpoint
    ckpt_path = args.model_path
    if not os.path.isfile(ckpt_path):
        logger.info(f"Error: Checkpoint not found at {ckpt_path}")
        return

    logger.info(f"Loading checkpoint: {ckpt_path}")
    # checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Handle state dict mismatch if necessary (e.g. module. prefix)
    state_dict = checkpoint['g_state']
    generator.load_state_dict(state_dict)
    
    # --- Initialize Test Data ---
    logger.info(f"Loading Test Data for {args.dataset}...")
    # Use data_loader with centered transformation (default to True if not in config)
    centered = getattr(args, 'centered', True)
    test_gen = data_loader(args, logger, split='test', phase='testing', centered=centered)
    if centered:
        logger.info("Scene-centered coordinate transformation enabled")
    else:
        logger.info("Scene-centered coordinate transformation disabled")
    
    if args.draw:
        draw_trajectory(args, test_gen, generator)
        return
    # --- Run Evaluation ---
    ade, fde = evaluate(args, test_gen, generator)
    
    logger.info("\n" + "="*30)
    logger.info(f"RESULTS for {args.dataset}")
    logger.info(f"Samples (K): {args.sample_k}")
    logger.info(f"ADE: {ade:.4f}")
    logger.info(f"FDE: {fde:.4f}")
    logger.info(f"ADE+FDE: {ade+fde:.4f}")
    logger.info("="*30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # --- Modified Arguments ---
    # input expected: path to the folder (e.g., "./results/1201_eth_orgLRs")
    parser.add_argument('--model_path', default=None, type=str, required=True, help='Path to model folder')
    parser.add_argument('--latest', default=False, action='store_true', help='Use latest checkpoint (default: best)') 
    parser.add_argument('--draw', default=False, action='store_true', help='Draw trajectory (default: False)')
    parser.add_argument('--draw_k', default=False, action='store_true', help='Draw all K trajectories in visualization (default: False, only draws best prediction)')
    
    # --- Evaluation Params ---
    # Dataset args (dataset, data_root_ethucy, data_root_nuscenes_pred) are loaded from config_saved.yaml
    parser.add_argument('--sample_k', default=20, type=int, help='Number of samples for Best-of-K evaluation')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for evaluation')
    parser.add_argument('--num_samples_check', default=None, type=int, help='Limit number of samples to evaluate (None = all)')
    parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data (default: False)')
    
    # Note: AgentFormer architecture params and data params are loaded from config_saved.yaml
    # Only keeping runtime/evaluation params here
    parser.add_argument('--use_gpu', default=0, type=int, help='Use GPU (0=CPU, 1=GPU)')
    
    args = parser.parse_args()

    model_dir = args.model_path
    args.output_dir = model_dir
    
    # 1. Get Folder Name (safely handling trailing slashes)
    folder_name = os.path.basename(os.path.normpath(model_dir))
    
    # 2. Construct Config Path
    config_path = os.path.join(model_dir, "config_saved.yaml")
    
    # 3. Construct Checkpoint Path
    # If --latest is set, look for "_latest.pt", otherwise "_best.pt"
    ckpt_suffix = "latest.pt" if args.latest else "best.pt"
    ckpt_filename = f"{folder_name}_{ckpt_suffix}"
    full_ckpt_path = os.path.join(model_dir, ckpt_filename)
    
    # Update args.model_path so main() loads the specific file
    args.model_path = full_ckpt_path
    
    print(f"Target Config: {config_path}")
    print(f"Target Checkpoint: {full_ckpt_path}")

    # --- Load Configuration ---
    # Preserve output_dir and use_gpu before loading config (they may contain old values from training)
    preserved_output_dir = args.output_dir
    preserved_use_gpu = args.use_gpu  # Preserve command-line value (default is 0)
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}. Cannot load dataset parameters.")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    # Load all config values into args (including architecture params not in parser)
    # We assume saved config is the "truth" for model architecture
    for key, value in cfg_dict.items():
        setattr(args, key, value)
    
    # Validate that required dataset parameters are loaded from config
    required_dataset_params = ['dataset', 'data_root_ethucy']
    missing_params = [param for param in required_dataset_params if not hasattr(args, param) or getattr(args, param) is None]
    if missing_params:
        print(f"Error: Required dataset parameters missing from config: {missing_params}")
        sys.exit(1)
    
    # Restore output_dir to the model directory (not the one from saved config)
    args.output_dir = preserved_output_dir
    
    # Restore use_gpu to command-line value (default is 0)
    # This prevents warnings when config has use_gpu=1 but CUDA is not available
    # use_gpu is a runtime setting, not a model architecture param, so we respect command line
    args.use_gpu = preserved_use_gpu

    # Add .get() compatibility
    args.get = lambda key, default=None: getattr(args, key, default)

    # Run Main
    main(args)