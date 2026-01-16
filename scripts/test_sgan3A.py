import argparse
import os
import sys
import yaml
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append(os.getcwd())
from model.data.dataloader import data_generator, data_loader
from model.sgan3A import AgentFormerGenerator
from utils.logger import Logger

# Import helper functions
from utils.train_helper import (
    get_device,
    prepare_batch,
    SmartBatcher,
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
    # Use SmartBatcher instead of fetch_and_collate_batch
    conn_dist = getattr(args, 'conn_dist', 100000.0)
    batcher = SmartBatcher(loader, batch_size=1, augment=False, max_agents_limit=50, conn_dist=conn_dist)
    batcher.reset()
    
    with torch.no_grad():
        while len(viz_samples) < 10 and batcher.has_data():
            raw_batch = batcher.next_batch()
            if raw_batch is None: continue
            
            batch = prepare_batch(raw_batch, device)
            pred_real_abs = batch['fut_motion'] # [Time, Agents, 2]
            
            # Generate K samples
            all_preds, _ = generator(batch, k=K) # Output: [agents, K, time, 2]
            all_preds = all_preds.permute(1, 2, 0, 3) # Stack: [K, Time, Agents, 2]
            
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
            gt_plot = np.vstack([last_obs, gt[:, n, :]])
            
            ax.plot(gt_plot[:, 0], gt_plot[:, 1], 'g-', linewidth=2, alpha=0.5, label='GT' if n==0 else "")
            ax.scatter(gt[-1, n, 0], gt[-1, n, 1], c='g', s=30, marker='*') # End marker
            if args.draw_pt:
                ax.scatter(gt[:, n, 0], gt[:, n, 1], c='g', s=12, alpha=0.6, marker='o')

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
            
            ax.plot(pred_plot[:, 0], pred_plot[:, 1], 'r', linewidth=2, color='darkred', label='Best Pred' if n==0 else "")
            if args.draw_pt:
                ax.scatter(best_pred[:, 0], best_pred[:, 1], c='darkred', s=12, alpha=0.6, marker='o')

        ax.set_title(f"Scene {i+1} (N={num_agents})")
        ax.axis('equal')
        if i == 0: ax.legend()

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, 'trajectory_viz.png')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Trajectory visualization saved to: {save_path}")


# def draw_trajectory_attention(args, loader, generator):
#     """
#     Visualizes 
#     """
#     logger.info("Drawing trajectories...")
#     generator.eval()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if args.shuffle:
#         loader.shuffle()
#     if hasattr(loader, 'reset'): loader.reset()
#     else: loader.index = 0
    
#     # Find a scene with 3 up agents
#     found_scene = False
#     batch = None
#     for raw_batch in loader:
#         if len(raw_batch['pre_motion_3D']) >= 3:
#             batch = prepare_batch(raw_batch, device)
#             found_scene = True
#             break
            
#     if not found_scene:
#         print("Cant find a scene with 3 up agents, terminate...")
#         return

#     # --Inference and find best--
#     protagonist_idx=0
#     with torch.no_grad():
#         K = args.sample_k if hasattr(args, 'sample_k') else 20
#         all_preds, data_dict = generator(batch, k=K, need_weights=True) # all_preds: [Agents, K, Time, 2]
        
#         pred_real = batch['fut_motion'] # [Time, Agents, 2]
#         obs_len = batch['pre_motion'].shape[0]
#         pred_len = pred_real.shape[0]
#         num_agents = batch['agent_num']

#         # Find best K (ADE) for protagonist
#         prot_gt = pred_real[:, protagonist_idx, :] # [Time, 2]
#         prot_preds = all_preds[protagonist_idx]    # [K, Time, 2]
#         ade_prot = torch.norm(prot_preds - prot_gt.unsqueeze(0), dim=-1).mean(dim=1) # [K]
#         best_k = torch.argmin(ade_prot).item()
        
#         # get attention weights for this best prediction (avg of the last layer of each Head)
#         # 根據你的輸出 shape (2, 20, 36, 24)，我們取 cross_attn_weights
#         cross_attn = data_dict['attn_weights']['cross_attn_weights'] 
#         # 如果有 K 維度則取 best_k，否則直接取最後一層平均
#         if cross_attn.ndim == 5: # [Batch/K, Layers, Heads, Tgt, Src]
#             attn_layer = cross_attn[best_k, -1].mean(dim=0).cpu().numpy()
#         else: # [Layers, Heads, Tgt, Src]
#             attn_layer = cross_attn[-1].mean(dim=0).cpu().numpy()


def draw_trajectory_attention(args, loader, generator):
    """
    Draw attention weights in the heat map in prediction space
    """
    logger.info("Drawing trajectories...")
    generator.eval()
    if args.shuffle:
        loader.shuffle()
    if hasattr(loader, 'reset'): loader.reset()
    else: loader.index = 0

    device = torch.device('cpu')  # Force CPU for testing
    
    # --- 1. 尋找 agent 數量 >= 3 的場景 ---
    conn_dist = getattr(args, 'conn_dist', 100000.0)
    batcher = SmartBatcher(loader, batch_size=1, augment=False, max_agents_limit=50, conn_dist=conn_dist)
    batcher.reset()

    found_scene = False
    batch = None
    
    # Find a scene with 3 up agents
    found_scene = False
    batch = None
    while batcher.has_data():
        raw_batch = batcher.next_batch()
        if raw_batch is None: continue
        if len(raw_batch['pre_motion_3D']) >= 3:
            batch = prepare_batch(raw_batch, device)
            found_scene = True
            break

    if not found_scene:
        print("Cant find a scene with 3 up agents, terminate...")
        return

    # --- 2. 推理與數據提取 ---
    with torch.no_grad():
        K = args.sample_k if hasattr(args, 'sample_k') else 10
        # 獲取權重時 need_weights=True
        all_preds, data_dict = generator(batch, k=K, need_weights=True)
        # all_preds: [Agents, K, Time, 2]
        num_agents = batch['agent_num']
        obs_len = batch['pre_motion'].shape[0]
        pred_len = all_preds.shape[2]
        protagonist_idx = 0

        # Compute best-k per agent (same as draw_trajectory)
        pred_real_abs = batch['fut_motion']
        all_preds_k = all_preds.permute(1, 2, 0, 3)  # [K, Time, Agents, 2]
        dist = torch.norm(all_preds_k - pred_real_abs.unsqueeze(0), dim=-1)
        ade_per_agent = dist.mean(dim=1)  # [K, Agents]
        best_indices = torch.argmin(ade_per_agent, dim=0)  # [Agents]

        # Use protagonist's best-k for attention visualization
        best_k = int(best_indices[protagonist_idx])

        # Extract cross-attention weights from the last decoder layer
        attn_w = data_dict['attn_weights']
        cross_attn_all = attn_w['cross_attn_weights'] # future-to-past/context attention weights
        self_attn_all = attn_w['self_attn_weights']  # future-to-future attention weights
        if cross_attn_all is None:
            print("No attention weights found, terminate...")
            return
        # select the best k if apply k-variety
        if cross_attn_all.ndim == 4:
            cross_attn = cross_attn_all[-1, best_k]  # [tgt_len, src_len]
        elif cross_attn_all.ndim == 3:
            cross_attn = cross_attn_all[-1]
        else:
            cross_attn = cross_attn_all
        if self_attn_all is None:
            print("No self-attention weights found, terminate...")
            return
        if self_attn_all.ndim == 4:
            self_attn = self_attn_all[-1, best_k]
        elif self_attn_all.ndim == 3:
            self_attn = self_attn_all[-1]
        else:
            self_attn = self_attn_all

    # --- 3. Plotting (1, 2) Subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_left, ax_right = axes

    past_coords = batch['pre_motion'].cpu().numpy()
    gt_coords = batch['fut_motion'].cpu().numpy()
    best_preds_all = np.stack(
        [all_preds[n, best_indices[n], :, :].cpu().numpy() for n in range(num_agents)],
        axis=0
    )

    # Left: predicted trajectories (same style as draw_trajectory)
    for n in range(num_agents):
        last_obs = past_coords[-1, n, :]
        ax_left.plot(
            past_coords[:, n, 0], past_coords[:, n, 1],
            'b-', linewidth=2, alpha=0.7, label='Past' if n == 0 else ""
        )
        ax_left.scatter(past_coords[0, n, 0], past_coords[0, n, 1], c='b', s=20, marker='o')

        gt_plot = np.vstack([last_obs, gt_coords[:, n, :]])
        ax_left.plot(
            gt_plot[:, 0], gt_plot[:, 1],
            'g-', linewidth=2, alpha=0.5, label='GT' if n == 0 else ""
        )
        ax_left.scatter(gt_coords[-1, n, 0], gt_coords[-1, n, 1], c='g', s=30, marker='*')

        pred_plot = np.vstack([last_obs, best_preds_all[n]])
        ax_left.plot(
            pred_plot[:, 0], pred_plot[:, 1],
            'r', linewidth=2, color='darkred', label='Best Pred' if n == 0 else ""
        )

    ax_left.set_title(f"Predicted Trajectories (N={num_agents})")
    ax_left.axis('equal')
    ax_left.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)

    # Right: attention heatmap scatter (past + forecast points)
    t_step = pred_len - 1
    tgt_seq_idx = t_step * num_agents + protagonist_idx
    src_len = cross_attn.shape[-1]
    obs_len_attn = src_len // num_agents if src_len % num_agents == 0 else obs_len
    obs_len_plot = min(obs_len, obs_len_attn)
    attn_flat_past = cross_attn[tgt_seq_idx]
    attn_past = attn_flat_past.reshape(obs_len_attn, num_agents)[:obs_len_plot, :]

    attn_flat_future = self_attn[tgt_seq_idx]
    attn_future = attn_flat_future.reshape(pred_len, num_agents)

    max_weight = max(float(attn_past.max()), float(attn_future.max()), 1e-8)
    attn_past = attn_past / max_weight
    attn_future = attn_future / max_weight

    # Draw trajectories and colored points
    for n in range(num_agents):
        traj_x = np.concatenate([past_coords[:, n, 0], best_preds_all[n, :, 0]])
        traj_y = np.concatenate([past_coords[:, n, 1], best_preds_all[n, :, 1]])
        ax_right.plot(traj_x, traj_y, color='gray', alpha=0.3, linewidth=1)

    past_plot = past_coords[:obs_len_plot]
    past_x = past_plot[:, :, 0].reshape(-1)
    past_y = past_plot[:, :, 1].reshape(-1)
    past_c = attn_past.reshape(-1)
    future_x = best_preds_all[:, :, 0].reshape(-1)
    future_y = best_preds_all[:, :, 1].reshape(-1)
    future_c = attn_future.T.reshape(-1)

    scatter_past = ax_right.scatter(past_x, past_y, c=past_c, cmap='Blues', s=60, alpha=0.9)
    scatter_future = ax_right.scatter(future_x, future_y, c=future_c, cmap='Greens', s=60, alpha=0.9)
    ax_right_pos = ax_right.get_position()
    bar_height = ax_right_pos.height * 0.03
    bar_width = ax_right_pos.width * 0.42
    bar_y0 = ax_right_pos.y1 + 0.02
    cax_past = fig.add_axes([
        ax_right_pos.x0 + ax_right_pos.width * 0.05,
        bar_y0,
        bar_width,
        bar_height
    ])
    cax_future = fig.add_axes([
        ax_right_pos.x0 + ax_right_pos.width * 0.53,
        bar_y0,
        bar_width,
        bar_height
    ])
    fig.colorbar(scatter_past, cax=cax_past, orientation='horizontal')
    fig.colorbar(scatter_future, cax=cax_future, orientation='horizontal')
    cax_past.set_xlabel('Past')
    cax_future.set_xlabel('Future')
    cax_past.xaxis.set_label_position('top')
    cax_future.xaxis.set_label_position('top')
    cax_past.tick_params(axis='x', labelsize=8, pad=1)
    cax_future.tick_params(axis='x', labelsize=8, pad=1)

    # Highlight protagonist's predicted last position
    ax_right.scatter(
        best_preds_all[protagonist_idx, t_step, 0],
        best_preds_all[protagonist_idx, t_step, 1],
        color='red', marker='*', s=180, zorder=5
    )

    ax_right.set_title(f"Cross-Attention Heatmap (Protagonist A{protagonist_idx})")
    attention_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='None', color='tab:blue', markersize=8, label='Attention to Past'),
        plt.Line2D([0], [0], marker='o', linestyle='None', color='tab:green', markersize=8, label='Attention to Future'),
        plt.Line2D([0], [0], marker='o', linestyle='None', color='red', markersize=8, label='Target (Being Predicted)')
    ]
    ax_right.legend(handles=attention_handles, loc='upper center', bbox_to_anchor=(0.5, 1.34), ncol=3, frameon=False)
    ax_right.set_aspect('equal', adjustable='box')

    fig.subplots_adjust(top=0.78, wspace=0.25)
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'trajectory_attention.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Visualization saved to: {save_path}")
    



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

    if args.draw_attn:
        draw_trajectory_attention(args, test_gen, generator)
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
    parser.add_argument('--draw_pt', default=False, action='store_true', help='Draw per-timestep points for GT and best prediction (default: False)')
    
    parser.add_argument('--draw_attn', default=False, action='store_true', help='Turn on mode to draw attention weights (default: False)')
    
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