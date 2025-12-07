#!/usr/bin/env python3
"""
Utility script to analyze and visualize training checkpoints.

This script extracts training metrics, losses, and other information
from checkpoint files saved during training.

Checkpoint files should be named: {folder_name}_latest.pt or {folder_name}_best.pt
and stored in a folder named {folder_name}.

Usage:
    python scripts/analyze_checkpoint.py --folder eth
    python scripts/analyze_checkpoint.py --folder eth --type best
    python scripts/analyze_checkpoint.py --folder checkpoints/eth
    python scripts/analyze_checkpoint.py --folder eth --no-plot  # Skip plotting
"""

import argparse
import torch
import os
import sys
from collections import defaultdict
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting disabled.")


def smooth_data(data, window_size=50):
    """
    Apply simple moving average smoothing to data.
    Uses a smaller window for the beginning values to avoid padding artifacts.
    
    Args:
        data: List or array of values to smooth
        window_size: Size of the moving average window
    
    Returns:
        Smoothed data array
    """
    if len(data) == 0:
        return np.array([])
    if len(data) < window_size:
        # For short data, use a smaller window
        window_size = max(1, len(data) // 2)
        if window_size < 2:
            return np.array(data)
    
    data_array = np.array(data)
    smoothed = []
    
    # Use progressively larger windows for the beginning
    for i in range(len(data_array)):
        if i < window_size:
            # Use smaller window for beginning
            w = min(i + 1, window_size)
            start = max(0, i - w + 1)
            smoothed.append(np.mean(data_array[start:i+1]))
        else:
            # Use full window
            smoothed.append(np.mean(data_array[i - window_size + 1:i+1]))
    
    return np.array(smoothed)


def has_metrics(checkpoint):
    """Check if checkpoint contains training metrics."""
    has_losses = ('G_losses' in checkpoint and checkpoint['G_losses']) or \
                 ('D_losses' in checkpoint and checkpoint['D_losses'])
    has_metrics = 'metrics_val' in checkpoint and checkpoint['metrics_val']
    has_timestamps = 'losses_ts' in checkpoint and checkpoint['losses_ts']
    return has_losses or has_metrics or has_timestamps


def load_checkpoint(checkpoint_path):
    """Load checkpoint from file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Check file extension
    if not checkpoint_path.endswith('.pt'):
        print(f"Warning: File does not have .pt extension: {checkpoint_path}")
    
    # Check file size (PyTorch checkpoints are usually > 1MB if they contain models)
    file_size = os.path.getsize(checkpoint_path)
    if file_size < 1024:  # Less than 1KB
        raise ValueError(f"File is too small ({file_size} bytes) to be a valid checkpoint.")
    
    print(f"Loading checkpoint from: {checkpoint_path} ({file_size / (1024*1024):.2f} MB)")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise ValueError(
            f"Failed to load checkpoint. The file may be corrupted or not a valid PyTorch checkpoint.\n"
            f"Original error: {str(e)}\n"
            f"Please ensure you're using a .pt file created by the training script."
        ) from e
    
    # Validate checkpoint structure
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dictionary. Invalid checkpoint format.")
    
    if 'args' not in checkpoint:
        print("Warning: Checkpoint does not contain 'args'. It may be an old format.")
    
    # Check if metrics are present
    if not has_metrics(checkpoint):
        print("Warning: Checkpoint does not appear to contain training metrics.")
        print("  This checkpoint may only contain model weights.")
        print("  Metrics are stored in checkpoints saved during training.")
    
    return checkpoint


def print_checkpoint_summary(checkpoint):
    """Print a summary of checkpoint contents."""
    print("\n" + "="*80)
    print("CHECKPOINT SUMMARY")
    print("="*80)
    
    # Training progress
    if 'counters' in checkpoint:
        counters = checkpoint['counters']
        print(f"\nTraining Progress:")
        print(f"  Current iteration (t): {counters.get('t', 'N/A')}")
        print(f"  Current epoch: {counters.get('epoch', 'N/A')}")
    
    # Best models
    if 'best_ade_fde' in checkpoint and checkpoint['best_ade_fde'] != float('inf'):
        print(f"  Best ADE+FDE: {checkpoint['best_ade_fde']:.4f}")
    if 'best_ade' in checkpoint and checkpoint['best_ade'] != float('inf'):
        print(f"  Best ADE: {checkpoint['best_ade']:.4f}")
    
    # Losses
    print(f"\nLosses recorded:")
    if 'G_losses' in checkpoint and checkpoint['G_losses']:
        print(f"  Generator losses: {list(checkpoint['G_losses'].keys())}")
        for k, v in checkpoint['G_losses'].items():
            if v:
                print(f"    {k}: {len(v)} values (last: {v[-1]:.4f})")
    
    if 'D_losses' in checkpoint and checkpoint['D_losses']:
        print(f"  Discriminator losses: {list(checkpoint['D_losses'].keys())}")
        for k, v in checkpoint['D_losses'].items():
            if v:
                print(f"    {k}: {len(v)} values (last: {v[-1]:.4f})")
    
    # Metrics
    print(f"\nValidation Metrics:")
    if 'metrics_val' in checkpoint and checkpoint['metrics_val']:
        for k, v in checkpoint['metrics_val'].items():
            if v:
                print(f"  {k}: {len(v)} values")
                print(f"    Latest: {v[-1]:.4f}")
                if len(v) > 1:
                    print(f"    Best: {min(v):.4f}")
                    print(f"    Worst: {max(v):.4f}")
    
    # Model states
    print(f"\nModel States:")
    has_model = 'g_state' in checkpoint and checkpoint['g_state'] is not None
    print(f"  Generator state: {'Present' if has_model else 'Not present (no_model checkpoint)'}")
    has_disc = 'd_state' in checkpoint and checkpoint['d_state'] is not None
    print(f"  Discriminator state: {'Present' if has_disc else 'Not present (no_model checkpoint)'}")
    
    # Training arguments
    if 'args' in checkpoint:
        print(f"\nTraining Arguments:")
        args = checkpoint['args']
        print(f"  Dataset: {args.get('dataset', 'N/A')}")
        print(f"  Batch size: {args.get('batch_size', 'N/A')}")
        print(f"  Learning rate (G): {args.get('g_learning_rate', 'N/A')}")
        print(f"  Learning rate (D): {args.get('d_learning_rate', 'N/A')}")
        print(f"  Use CVAE: {args.get('use_cvae', 'N/A')}")
        print(f"  Encoder layers: {args.get('enc_layers', 'N/A')}")
        print(f"  Decoder layers: {args.get('dec_layers', 'N/A')}")


def plot_training_curves(checkpoint, output_dir=None, smooth_window=5):
    """
    Plot training curves in a 1x3 horizontal layout.
    
    Layout:
    - Left: Validation Accuracy (ADE/FDE)
    - Middle: Smoothed Adversarial Losses (G_adv and D_loss)
    - Right: G L2 Loss (Training vs Validation)
    
    Args:
        checkpoint: Checkpoint dictionary
        output_dir: Directory to save plots (None to display)
        smooth_window: Window size for smoothing curves
    """
    if not PLOTTING_AVAILABLE:
        print("Cannot plot: matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Training Diagnostics Dashboard', fontsize=16, fontweight='bold')
    
    # Get timestamps
    losses_ts = checkpoint.get('losses_ts', [])
    sample_ts = checkpoint.get('sample_ts', [])
    
    # ============================================================================
    # Left: Validation Accuracy (Meters)
    # Plot: val_ade, val_fde
    # ============================================================================
    ax1 = axes[0]
    if 'metrics_val' in checkpoint and sample_ts:
        metrics_val = checkpoint['metrics_val']
        
        if 'ade' in metrics_val and metrics_val['ade']:
            min_len = min(len(metrics_val['ade']), len(sample_ts))
            ax1.plot(sample_ts[:min_len], metrics_val['ade'][:min_len], 
                    label='Val ADE', marker='s', markersize=5, linewidth=2, color='#1f77b4')
        
        if 'fde' in metrics_val and metrics_val['fde']:
            min_len = min(len(metrics_val['fde']), len(sample_ts))
            ax1.plot(sample_ts[:min_len], metrics_val['fde'][:min_len], 
                    label='Val FDE', marker='^', markersize=5, linewidth=2, color='#ff7f0e')
    
    # ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_xlabel('iteration', fontsize=11)
    ax1.set_ylabel('Error (meters)', fontsize=11)
    ax1.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ============================================================================
    # Middle: Smoothed Adversarial Losses
    # Plot: Smoothed G_adv and D_loss
    # ============================================================================
    ax2 = axes[1]
    if 'G_losses' in checkpoint and 'D_losses' in checkpoint and losses_ts:
        g_losses = checkpoint['G_losses']
        d_losses = checkpoint['D_losses']
        
        # Plot smoothed G adversarial loss
        if 'G_adv' in g_losses and g_losses['G_adv']:
            min_len = min(len(g_losses['G_adv']), len(losses_ts))
            g_adv_data = g_losses['G_adv'][:min_len]
            g_adv_smoothed = smooth_data(g_adv_data, window_size=smooth_window)
            ax2.plot(losses_ts[:min_len], g_adv_smoothed, 
                    label='G Adversarial (smoothed)', linewidth=2.5, color='#2ca02c', alpha=0.8)
            # Also plot raw data with lower opacity
            ax2.plot(losses_ts[:min_len], g_adv_data, 
                    linewidth=0.5, color='#2ca02c', alpha=0.2)
            
            # Plot min/max range if available
            if 'G_adv_min' in g_losses and 'G_adv_max' in g_losses:
                g_adv_min = g_losses['G_adv_min'][:min_len]
                g_adv_max = g_losses['G_adv_max'][:min_len]
                ax2.fill_between(losses_ts[:min_len], g_adv_min, g_adv_max, 
                               color='#2ca02c', alpha=0.15, label='G Adv range')
        
        # Plot smoothed D loss
        if 'D_loss' in d_losses and d_losses['D_loss']:
            min_len = min(len(d_losses['D_loss']), len(losses_ts))
            d_loss_data = d_losses['D_loss'][:min_len]
            d_loss_smoothed = smooth_data(d_loss_data, window_size=smooth_window)
            ax2.plot(losses_ts[:min_len], d_loss_smoothed, 
                    label='D Loss (smoothed)', linewidth=2.5, color='#ff7f0e', alpha=0.8)
            # Also plot raw data with lower opacity
            ax2.plot(losses_ts[:min_len], d_loss_data, 
                    linewidth=0.5, color='#ff7f0e', alpha=0.2)
            
            # Plot min/max range if available
            if 'D_loss_min' in d_losses and 'D_loss_max' in d_losses:
                d_loss_min = d_losses['D_loss_min'][:min_len]
                d_loss_max = d_losses['D_loss_max'][:min_len]
                ax2.fill_between(losses_ts[:min_len], d_loss_min, d_loss_max, 
                               color='#ff7f0e', alpha=0.15, label='D Loss range')
    
    # ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_xlabel('iteration', fontsize=11)
    ax2.set_ylabel('Adversarial Loss', fontsize=11)
    ax2.set_title('Smoothed Adversarial Losses', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # ============================================================================
    # Right: G L2 Loss (Training vs Validation)
    # Plot: Training and Validation G L2 loss (raw and smoothed)
    # ============================================================================
    ax3 = axes[2]
    has_data = False
    
    # Training G L2 loss
    if 'G_losses' in checkpoint and losses_ts:
        g_losses = checkpoint['G_losses']
        
        if 'G_l2' in g_losses and g_losses['G_l2']:
            min_len = min(len(g_losses['G_l2']), len(losses_ts))
            train_l2_data = g_losses['G_l2'][:min_len]
            train_l2_smoothed = smooth_data(train_l2_data, window_size=smooth_window)
            
            # Plot smoothed training L2 loss
            ax3.plot(losses_ts[:min_len], train_l2_smoothed, 
                    label='Train G L2 (smoothed)', linewidth=2.5, color='#1f77b4', alpha=0.8)
            # Plot raw training L2 loss with lower opacity
            ax3.plot(losses_ts[:min_len], train_l2_data, 
                    linewidth=0.5, color='#1f77b4', alpha=0.2)
            
            # # Plot min/max range if available
            # if 'G_l2_min' in g_losses and 'G_l2_max' in g_losses:
            #     train_l2_min = g_losses['G_l2_min'][:min_len]
            #     train_l2_max = g_losses['G_l2_max'][:min_len]
            #     ax3.fill_between(losses_ts[:min_len], train_l2_min, train_l2_max, 
            #                    color='#1f77b4', alpha=0.15, label='Train L2 range')
            has_data = True
    
    # Validation G L2 loss
    if 'metrics_val' in checkpoint and sample_ts:
        metrics_val = checkpoint['metrics_val']
        
        if 'l2' in metrics_val and metrics_val['l2']:
            min_len = min(len(metrics_val['l2']), len(sample_ts))
            val_l2_data = metrics_val['l2'][:min_len]
            val_l2_smoothed = smooth_data(val_l2_data, window_size=smooth_window)
            
            # Plot smoothed validation L2 loss
            ax3.plot(sample_ts[:min_len], val_l2_smoothed, 
                    label='Val G L2 (smoothed)', linewidth=2.5, color='#d62728', alpha=0.8, linestyle='--')
            # Plot raw validation L2 loss with lower opacity
            ax3.plot(sample_ts[:min_len], val_l2_data, 
                    linewidth=0.5, color='#d62728', alpha=0.2, linestyle='--')
            has_data = True
    
    if not has_data:
        ax3.text(0.5, 0.5, 'No L2 loss data available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    
    # ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_xlabel('iteration', fontsize=11)
    ax3.set_ylabel('L2 Loss', fontsize=11)
    ax3.set_title('G L2 Loss (Training vs Validation)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    # if has_data:
        # Calculate y_max from available data (including max values if available)
        # max_values = []
        # if 'G_losses' in checkpoint and losses_ts and 'G_l2' in checkpoint['G_losses']:
        #     train_l2_data = checkpoint['G_losses']['G_l2']
        #     if len(train_l2_data) > 0:
        #         max_values.append(np.max(train_l2_data))
        #         # Also check for G_l2_max if available
        #         if 'G_l2_max' in checkpoint['G_losses']:
        #             train_l2_max = checkpoint['G_losses']['G_l2_max']
        #             if len(train_l2_max) > 0:
        #                 max_values.append(np.max(train_l2_max))
        # if 'metrics_val' in checkpoint and sample_ts and 'l2' in checkpoint['metrics_val']:
        #     val_l2_data = checkpoint['metrics_val']['l2']
        #     if len(val_l2_data) > 0:
        #         max_values.append(np.max(val_l2_data))
        # if max_values:
        #     y_max = np.max(max_values) * 1.1  # Add 10% padding
        #     ax4.set_ylim([0, y_max])
    #     else:
    #         ax4.set_ylim(bottom=0)
    # else:
    #     ax4.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {plot_path}")
    else:
        plt.show()


def export_to_json(checkpoint, output_path):
    """Export checkpoint metrics to JSON format."""
    export_data = {
        'counters': checkpoint.get('counters', {}),
        'best_ade': checkpoint.get('best_ade', float('inf')),
        'best_ade_fde': checkpoint.get('best_ade_fde', float('inf')),
        'losses_ts': checkpoint.get('losses_ts', []),
        'sample_ts': checkpoint.get('sample_ts', []),
    }
    
    # Convert defaultdict to regular dict for JSON
    if 'G_losses' in checkpoint:
        export_data['G_losses'] = dict(checkpoint['G_losses'])
    if 'D_losses' in checkpoint:
        export_data['D_losses'] = dict(checkpoint['D_losses'])
    if 'metrics_val' in checkpoint:
        export_data['metrics_val'] = dict(checkpoint['metrics_val'])
    
    # Convert numpy/torch types to native Python types
    def convert_to_native(obj):
        if isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif hasattr(obj, 'item'):  # torch.Tensor
            return obj.item()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, float) and (obj == float('inf') or obj == float('-inf')):
            return str(obj)  # Convert inf to string for JSON compatibility
        else:
            return obj
    
    export_data = convert_to_native(export_data)
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\nExported metrics to JSON: {output_path}")


def find_checkpoint_path(folder_name, checkpoint_type='latest'):
    """
    Find checkpoint path given folder name and type.
    
    Args:
        folder_name: Name of the folder (can be full path or just name)
        checkpoint_type: 'latest' or 'best' (default: 'latest')
    
    Returns:
        Path to the checkpoint file
    """
    # Normalize the folder name (remove trailing slashes)
    folder_name = folder_name.rstrip('/')
    
    # Check if it's an absolute path or exists as-is
    if os.path.isdir(folder_name):
        folder_path = os.path.abspath(folder_name)
        folder_basename = os.path.basename(folder_path)
    else:
        # Try relative to current directory
        if os.path.exists(folder_name):
            folder_path = os.path.abspath(folder_name)
            folder_basename = os.path.basename(folder_path)
        else:
            # Try in checkpoints directory
            checkpoints_dir = os.path.join(os.getcwd(), 'checkpoints')
            potential_path = os.path.join(checkpoints_dir, folder_name)
            if os.path.exists(potential_path):
                folder_path = os.path.abspath(potential_path)
                folder_basename = folder_name
            else:
                # Try as absolute path one more time
                if os.path.isabs(folder_name) and os.path.exists(folder_name):
                    folder_path = folder_name
                    folder_basename = os.path.basename(folder_path)
                else:
                    # Assume it's a relative path that should exist
                    folder_path = os.path.abspath(folder_name)
                    folder_basename = os.path.basename(folder_path)
    
    # Construct checkpoint filename: {folder_name}_{latest|best}.pt
    checkpoint_filename = f"{folder_basename}_{checkpoint_type}.pt"
    checkpoint_path = os.path.join(folder_path, checkpoint_filename)
    
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description='Analyze training checkpoints and extract metrics'
    )
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Folder name containing checkpoints. Checkpoint files should be named '
             '{folder_name}_latest.pt or {folder_name}_best.pt. '
             'Can be a folder name (e.g., "eth") or full path (e.g., "checkpoints/eth")'
    )
    parser.add_argument(
        '--type',
        type=str,
        choices=['latest', 'best'],
        default='latest',
        help='Type of checkpoint to load: "latest" or "best" (default: latest)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating training curve plots (plots are generated by default)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: same as checkpoint folder)'
    )
    parser.add_argument(
        '--export-json',
        type=str,
        default=None,
        help='Export metrics to JSON file (optional)'
    )
    parser.add_argument(
        '--smooth-window',
        type=int,
        default=1,
        help='Window size for smoothing curves (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Find checkpoint path
    checkpoint_path = find_checkpoint_path(args.folder, args.type)
    
    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_path):
        # Try to provide helpful error message
        folder_path = os.path.dirname(checkpoint_path)
        if not os.path.exists(folder_path):
            print(f"Error: Folder not found: {folder_path}")
            print(f"  Searched for: {args.folder}")
            print(f"  Expected checkpoint: {checkpoint_path}")
        else:
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            print(f"  Folder exists: {folder_path}")
            # List available files in the folder
            if os.path.isdir(folder_path):
                available_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
                if available_files:
                    print(f"  Available checkpoint files in folder:")
                    for f in available_files:
                        print(f"    - {f}")
                else:
                    print(f"  No .pt files found in folder")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Check if metrics are present before proceeding
    if not has_metrics(checkpoint):
        print("\n" + "="*80)
        print("WARNING: This checkpoint does not contain training metrics.")
        print("The checkpoint may only contain model weights.")
        print("To analyze training progress, use a checkpoint saved during training.")
        print("="*80)
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
    
    # Print summary
    print_checkpoint_summary(checkpoint)
    
    # Plot by default (unless --no-plot is specified)
    if not args.no_plot:
        if not has_metrics(checkpoint):
            print("\nCannot generate plots: checkpoint does not contain metrics.")
        else:
            output_dir = args.output_dir or os.path.dirname(checkpoint_path)
            plot_training_curves(checkpoint, output_dir, smooth_window=args.smooth_window)
    
    # Export to JSON if requested
    if args.export_json:
        if not has_metrics(checkpoint):
            print("\nCannot export metrics: checkpoint does not contain metrics.")
        else:
            export_to_json(checkpoint, args.export_json)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
