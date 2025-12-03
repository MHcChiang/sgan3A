import argparse
import os
import sys
from torch.distributed import batch_isend_irecv
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
from model.data.dataloader import data_generator
from model.sgan3A import AgentFormerGenerator
from utils.logger import Logger

# ==========================================
# 1. Helpers (Reused from train_sgan3A.py)
# ==========================================
def get_device(args):
    return torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')


def prepare_batch(batch, device):
    def to_tensor(x):
        if isinstance(x, torch.Tensor): return x.clone().detach().float()
        elif isinstance(x, np.ndarray): return torch.from_numpy(x).float()
        else: return torch.tensor(x).float()

    data = {}
    pre_motion = torch.stack([to_tensor(m) for m in batch['pre_motion_3D']], dim=0).to(device)
    pre_motion = pre_motion.transpose(0, 1).contiguous()
    
    fut_motion = torch.stack([to_tensor(m) for m in batch['fut_motion_3D']], dim=0).to(device)
    fut_motion = fut_motion.transpose(0, 1).contiguous()

    data['pre_motion'] = pre_motion
    data['fut_motion'] = fut_motion
    data['agent_num'] = pre_motion.shape[1]
    
    if 'agent_mask' in batch:
        data['agent_mask'] = to_tensor(batch['agent_mask']).to(device)
    else:
        data['agent_mask'] = torch.zeros(data['agent_num'], data['agent_num']).to(device)

    data['heading'] = torch.zeros(data['agent_num']).to(device)
    data['heading_vec'] = torch.zeros(data['agent_num'], 2).to(device)
    data['pre_vel'] = torch.zeros_like(pre_motion)
    data['pre_motion_scene_norm'] = pre_motion
    data['agent_enc_shuffle'] = None
    
    if 'fut_motion_mask' in batch:
        mask = torch.stack([to_tensor(m) for m in batch['fut_motion_mask']], dim=0).to(device)
        data['fut_mask'] = mask.transpose(0, 1).contiguous()
    
    return data

# ==========================================
# 2. Evaluation Loop
# ==========================================
def evaluate(args, loader, generator):
    ade_outer, fde_outer = [], []
    total_traj = 0
    device = get_device(args)
    
    generator.eval()
    if hasattr(loader, 'reset'): loader.reset()
    else: loader.index = 0
    
    K = args.sample_k 
    logger.info(f"Starting Evaluation (Best-of-{K})...")
    
    i = 1
    with torch.no_grad():
        while not loader.is_epoch_end():
            # 1. Fetch One Sample
            raw_batch = loader()
            if raw_batch is None: continue
            
            # 2. Convert
            batch = prepare_batch(raw_batch, device)
            
            pred_real_abs = batch['fut_motion'] # [Time, Agents, 2]
            
            # 3. Generate K Samples
            batch_samples = []
            for _ in range(K):
                # Generate
                pred_fake, _ = generator(batch) 
                
                # Permute: [Agents, Time] -> [Time, Agents] to match Real
                pred_fake = pred_fake.permute(1, 0, 2)
                batch_samples.append(pred_fake.unsqueeze(0)) 
                
            # Stack K samples: [K, Time, Agents, 2]
            all_preds = torch.cat(batch_samples, dim=0)
            
            # 4. Compute Metrics
            # Expand Ground Truth: [1, Time, Agents, 2]
            gt_expanded = pred_real_abs.unsqueeze(0)
            
            # Distance: [K, Time, Agents]
            diff = all_preds - gt_expanded
            dist = torch.norm(diff, dim=-1)
            
            # Metrics per sample
            ade_per_sample = dist.mean(dim=1) # [K, Agents]
            fde_per_sample = dist[:, -1, :]   # [K, Agents]
            # breakpoint()
            # Handle Masks
            if 'fut_mask' in batch:
                valid_mask = batch['fut_mask'] > 0 # [Time, Agents]
                valid_counts = valid_mask.sum(dim=0).unsqueeze(0) # [1, Agents]
                
                masked_dist = dist * valid_mask.unsqueeze(0)
                ade_per_sample = masked_dist.sum(dim=1) / (valid_counts + 1e-6)
                
                last_valid = valid_mask[-1].unsqueeze(0)
                fde_per_sample = fde_per_sample * last_valid

            # Select Best Sample (Min Error)
            min_ade, _ = ade_per_sample.min(dim=0) # [Agents]
            min_fde, _ = fde_per_sample.min(dim=0) # [Agents]
            print(f"scene: {i}, ade: {min_ade}, fde: {min_fde}")
            ade_outer.append(min_ade)
            fde_outer.append(min_fde)
            total_traj += batch['agent_num']

    if len(ade_outer) > 0:
        ade_score = torch.cat(ade_outer).mean().item()
        fde_score = torch.cat(fde_outer).mean().item()
    else:
        ade_score, fde_score = 0.0, 0.0
    
    return ade_score, fde_score


def draw_trajectory(args, loader, generator):
    """
    Visualizes 10 sequences with "Per-Agent Best" logic.
    - Past: Blue
    - GT Future: Green
    - All K Predictions: Light Red (Thin)
    - Best Prediction (Per Agent): Dark Red (Bold)
    """
    logger.info("Drawing trajectories...")
    generator.eval()
    # loader.shuffle()
    if hasattr(loader, 'reset'): loader.reset()
    else: loader.index = 0
    
    viz_samples = []
    device = get_device(args)
    K = args.sample_k
    
    # --- 1. Collect 10 Samples ---
    with torch.no_grad():
        while len(viz_samples) < 10 and not loader.is_epoch_end():
            raw_batch = loader()
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

            # --- C. Plot Best Prediction (Red) ---
            # FIX 1: Draw ONLY the best trajectory
            best_k = best_indices[n]
            best_pred = all_preds[best_k, :, n, :] # [Time, 2]
            
            # FIX 2: Fill Gap
            pred_plot = np.vstack([last_obs, best_pred])
            
            ax.plot(pred_plot[:, 0], pred_plot[:, 1], 'r--', linewidth=2, label='Pred' if n==0 else "")

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
    
    device = get_device(args)
    logger.info(f'Using device: {device}')

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
    test_gen = data_generator(args, logger, split='test', phase='testing')
    
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
    logger.info("="*30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # --- Modified Arguments ---
    # input expected: path to the folder (e.g., "./results/1201_eth_orgLRs")
    parser.add_argument('--model_path', default=None, type=str, required=True, help='Path to model folder')
    parser.add_argument('--latest', default=False, action='store_true', help='Use latest checkpoint (default: best)')
    parser.add_argument('--draw', default=False, action='store_true', help='Draw trajectory (default: False)')
    
    # --- Evaluation Params ---
    parser.add_argument('--dataset', default='eth', type=str)
    parser.add_argument('--data_root_ethucy', default='datasets/eth_ucy', type=str)
    parser.add_argument('--data_root_nuscenes_pred', default='datasets/nuscenes_pred', type=str)
    parser.add_argument('--sample_k', default=20, type=int, help='Number of samples for Best-of-K evaluation')
    parser.add_argument('--batch_size', default=8, type=int)
    
    # --- AgentFormer Params (Defaults will be overwritten by config_saved.yaml) ---
    parser.add_argument('--traj_scale', default=1, type=int)
    parser.add_argument('--motion_dim', default=2, type=int)
    parser.add_argument('--forecast_dim', default=2, type=int)
    parser.add_argument('--tf_model_dim', default=256, type=int)
    parser.add_argument('--tf_nhead', default=8, type=int)
    parser.add_argument('--tf_ff_dim', default=512, type=int)
    parser.add_argument('--tf_dropout', default=0.1, type=float)
    parser.add_argument('--pos_concat', action='store_true', default=True)
    parser.add_argument('--nz', default=32, type=int)
    parser.add_argument('--z_type', default='gaussian', type=str)
    
    parser.add_argument('--enc_layers', default=2, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--input_type', default='pos', type=str)
    parser.add_argument('--use_cvae', default=0, type=int)
    parser.add_argument('--use_map', default=False, action='store_true')

    # Misc
    parser.add_argument('--past_frames', default=8, type=int)
    parser.add_argument('--future_frames', default=12, type=int)
    parser.add_argument('--min_past_frames', default=8, type=int)
    parser.add_argument('--min_future_frames', default=12, type=int)
    parser.add_argument('--frame_skip', default=1, type=int)
    # parser.add_argument('--output_dir', default='./results/test_output')
    parser.add_argument('--use_gpu', default=1, type=int)
    
    args = parser.parse_args()

    # --- NEW: Logic to resolve paths from Folder ---
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
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        
        # Overwrite args with saved config values
        # We assume saved config is the "truth" for model architecture
        for key, value in cfg_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)
    else:
        print(f"Warning: Config file not found at {config_path}. Using default args.")

    # Add .get() compatibility
    args.get = lambda key, default=None: getattr(args, key, default)

    # Run Main
    main(args)