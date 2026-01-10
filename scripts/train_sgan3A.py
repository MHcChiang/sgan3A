import argparse
import gc
import logging
import os
import sys
import time
import numpy as np
from collections import defaultdict, deque
import yaml
import copy

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append(os.getcwd())
# Add scripts directory to path for importing analyze_checkpoint
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Import Data Generator (The Iterator based one)
from model.data.dataloader import data_generator, data_loader
from model.sgan3A import AgentFormerGenerator, AgentFormerDiscriminator
from model.losses import gan_g_loss, gan_d_loss, l2_loss, select_best_k_scene
from model.losses import g_hinge_loss, d_hinge_loss # new
from utils.logger import Logger
from utils.train_helper import (
    init_weights,
    get_device,
    create_scheduler,
    collate_scenes,
    SmartBatcher,
    prepare_batch,
    rotate_scene,
    relative_to_abs,
    check_accuracy,
    load_config_from_yaml,
)

# Import plotting function from analyze_checkpoint
try:
    from analyze_checkpoint import plot_training_curves
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

torch.backends.cudnn.benchmark = True

# ==========================================
# Argument Parser
# ==========================================

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# ---Configuration file---
parser.add_argument('--cfg', default=None, type=str, help='Path to .yml config file')

# --- Data Params ---
parser.add_argument('--dataset', default='eth', type=str)
parser.add_argument('--data_root_ethucy', default='datasets/eth_ucy', type=str)
parser.add_argument('--traj_scale', default=1, type=int)
parser.add_argument('--past_frames', default=8, type=int)
parser.add_argument('--future_frames', default=12, type=int)
parser.add_argument('--min_past_frames', default=8, type=int)
parser.add_argument('--min_future_frames', default=12, type=int)
parser.add_argument('--frame_skip', default=1, type=int)
parser.add_argument('--phase', default='training', type=str)
parser.add_argument('--augment', default=0, type=int, help='1 to enable data augmentation, 0 for no augmentation')
parser.add_argument('--centered', default=1, type=int, help='1 to enable scene-centered coordinate transformation, 0 to disable')
parser.add_argument('--conn_dist', default=100000.0, type=float, help='Distance threshold for agent attention (valid value: below 10000.0)')

# --- Optimization ---
parser.add_argument('--batch_size', default=8, type=int) # Scenes per batch (if collating) or just 1
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--best_k', default=10, type=int, help='Number of samples for Variety Loss')

parser.add_argument('--g_learning_rate', default=1e-4, type=float)
parser.add_argument('--d_learning_rate', default=1e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--clipping_threshold_d', default=0, type=float)
parser.add_argument('--k', default=1, type=int, help='Number of samples for Variety Loss')
parser.add_argument('--warmup_epochs', default=3, type=int, help='Number of epochs to train G only on L2 loss')
parser.add_argument('--resume_warmup_from', default=None, type=str, help='Path to pre-trained warm-up model')

# --- Learning Rate Scheduler ---
parser.add_argument('--scheduler_type', default='none', type=str,
                    choices=['none', 'step', 'exponential', 'plateau', 'cosine'],
                    help='Type of learning rate scheduler (applied to both G and D)')
parser.add_argument('--scheduler_step_size', default=None, type=int,
                    help='Period of learning rate decay for StepLR (epochs). If None, uses 30%% of total epochs')
parser.add_argument('--scheduler_gamma', default=None, type=float,
                    help='Multiplicative factor for StepLR/ExponentialLR decay. If None, uses 0.1 for StepLR, 0.95 for Exponential')
parser.add_argument('--scheduler_min_lr', default=0, type=float,
                    help='Minimum learning rate for CosineAnnealingLR (default: 0)')
parser.add_argument('--scheduler_patience', default=10, type=int,
                    help='Number of epochs with no improvement for ReduceLROnPlateau (default: 10)')
parser.add_argument('--scheduler_factor', default=0.5, type=float,
                    help='Factor by which LR is reduced for ReduceLROnPlateau (default: 0.5)')

# --- AgentFormer Params ---
parser.add_argument('--motion_dim', default=2, type=int)
parser.add_argument('--forecast_dim', default=2, type=int)
parser.add_argument('--tf_model_dim', default=256, type=int) # Paper: 256 [cite: 280]
parser.add_argument('--tf_nhead', default=8, type=int)       # Paper: 8 heads [cite: 280]
parser.add_argument('--tf_ff_dim', default=512, type=int)    # Paper: 512 hidden units [cite: 280]
parser.add_argument('--tf_dropout', default=0.1, type=float) # Paper: 0.1 dropout [cite: 279]
parser.add_argument('--pos_concat', action='store_true', default=True)
parser.add_argument('--nz', default=32, type=int)            # Paper: latent dim 32 [cite: 281]
parser.add_argument('--z_type', default='gaussian', type=str)
parser.add_argument('--enc_layers', default=2, type=int, help='Layers in Context Encoder')
parser.add_argument('--dec_layers', default=2, type=int, help='Layers in Future Decoder')
parser.add_argument('--input_type', default='pos', type=str) # 'pos' or 'vel'
parser.add_argument('--noise_std', default=0.0, type=float, help='Add Gaussian noise for future trajectories in Discriminator(default: 0.0). Will decay in 20 epochs to 0.05.')

# --- CVAE Control ---
parser.add_argument('--use_cvae', default=0, type=int, help='1 to enable CVAE (KL loss), 0 for Pure GAN')
parser.add_argument('--kl_weight', default=1.0, type=float)
parser.add_argument('--l2_loss_weight', default=1.0, type=float)

# --- Output ---
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=50, type=int)
parser.add_argument('--checkpoint_every', default=10, type=int) # Checkpoint often
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--restore_from_checkpoint', default=0, type=int)
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--num_samples_check', default=500, type=int)
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--gpu_num', default="0", type=str)

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, 'log.txt')
    logger = Logger(log_path)

    # Save the final merged configuration to the output folder for reproducibility
    config_save_path = os.path.join(args.output_dir, 'config_saved.yaml')
    # with open(config_save_path, 'w') as f:
    #     yaml.dump(vars(args), f, default_flow_style=False)
    # Create a clean dictionary without functions
    save_dict = vars(args).copy()
    if 'get' in save_dict:
        del save_dict['get']
        
    with open(config_save_path, 'w') as f:
        yaml.dump(save_dict, f, default_flow_style=False)
    logger.info(f'Configuration saved to: {config_save_path}')

    device = get_device(args)
    args.device = device
    logger.info(f'Using device: {device}')
    
    # Verify GPU is working if CUDA is selected
    if device.type == 'cuda':
        logger.info(f'CUDA Device: {torch.cuda.get_device_name(device.index)}')
        logger.info(f'CUDA Available: {torch.cuda.is_available()}')
        logger.info(f'CUDA Device Count: {torch.cuda.device_count()}')
        # Test GPU with a simple operation
        try:
            test_tensor = torch.zeros(1).to(device)
            logger.info(f'GPU test successful: {test_tensor.device}')
        except Exception as e:
            logger.error(f'GPU test failed: {e}')
            logger.warning('Falling back to CPU')
            device = torch.device('cpu')
            args.device = device
    else:
        logger.info('Using CPU (CUDA not available or use_gpu=0)')

    # --- Initialize Models ---
    args.context_encoder = {
        'nlayer': args.enc_layers,
        'input_type': [args.input_type] # e.g. ['pos']
    }
    # args to pass to future decoder in agentformer.py
    args.future_decoder = {
        'nlayer': args.dec_layers,
        'out_mlp_dim': [256, 128], # Standard MLP head dims
        'input_type': [args.input_type]
    }
    
    # Also needed if using CVAE mode
    # args.future_encoder = {
    #     'nlayer': args.enc_layers,
    #     'out_mlp_dim': [512, 256],
    #     'input_type': [args.input_type]
    # }

    # We pass 'args' as 'cfg' to the models
    generator = AgentFormerGenerator(args).to(device)
    generator.apply(init_weights)
    
    discriminator = AgentFormerDiscriminator(args).to(device)
    discriminator.apply(init_weights)

    logger.info("------------- Generator Architecture -------------")
    logger.info(str(generator))
    logger.info("--------------------------------------------------")

    logger.info("----------- Discriminator Architecture -----------")
    logger.info(str(discriminator))
    logger.info("--------------------------------------------------")

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)

    # --- Initialize Learning Rate Schedulers ---
    scheduler_g = create_scheduler(optimizer_g, args.scheduler_type, args.num_epochs, args)
    scheduler_d = create_scheduler(optimizer_d, args.scheduler_type, args.num_epochs, args)

    # scaler = torch.amp.GradScaler(args.device)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    # scaler = torch.cuda.amp.GradScaler(enabled=(args.device.type == 'cuda'))
    
    if scheduler_g is not None:
        logger.info(f'Using scheduler: {args.scheduler_type} (applied to both G and D)')

    # --- Initialize Metrics Storage ---
    checkpoint = {
        'args': None, # Will be filled later
        'G_losses': defaultdict(list),
        'D_losses': defaultdict(list),
        'losses_ts': [],
        'metrics_val': defaultdict(list),
        'metrics_train': defaultdict(list),
        'sample_ts': [],
        'counters': {'t': None, 'epoch': None},
        'g_state': None,
        'g_optim_state': None,
        'g_scheduler_state': None,
        'd_state': None,
        'd_optim_state': None,
        'd_scheduler_state': None,
        'best_ade': float('inf'),  # Keep for backward compatibility
        'best_ade_fde': float('inf'),  # ADE + FDE for best model selection
    }

    # --- Restore Checkpoint ---
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir, f'{args.checkpoint_name}_latest.pt')

    
    if restore_path is not None and os.path.isfile(restore_path):
        logger.info(f'Restoring from checkpoint {restore_path}')
        checkpoint = torch.load(restore_path, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        
        # Restore scheduler states if available
        if scheduler_g is not None and 'g_scheduler_state' in checkpoint and checkpoint['g_scheduler_state'] is not None:
            scheduler_g.load_state_dict(checkpoint['g_scheduler_state'])
            logger.info('Restored generator scheduler state')
        if scheduler_d is not None and 'd_scheduler_state' in checkpoint and checkpoint['d_scheduler_state'] is not None:
            scheduler_d.load_state_dict(checkpoint['d_scheduler_state'])
            logger.info('Restored discriminator scheduler state')
        
        t = checkpoint['counters']['t'] if checkpoint['counters']['t'] is not None else 0
        epoch = checkpoint['counters']['epoch'] if checkpoint['counters']['epoch'] is not None else 0
        best_ade = checkpoint.get('best_ade', float('inf'))  # Keep for backward compatibility
        best_ade_fde = checkpoint.get('best_ade_fde', float('inf'))
    else:
        logger.info('Starting new training')
        t = 0
        epoch = 0
        best_ade = float('inf') # Keep for backward compatibility
        best_ade_fde = float('inf')
    
    if args.resume_warmup_from and os.path.isfile(args.resume_warmup_from):
        logger.info(f"Loading pre-trained warm-up model from: {args.resume_warmup_from}")
        warm_checkpoint = torch.load(args.resume_warmup_from, map_location='cpu', weights_only=False)
    
        generator.load_state_dict(warm_checkpoint['g_state'])
        discriminator.load_state_dict(warm_checkpoint['d_state'])
        del warm_checkpoint
    
        args.warmup_epochs = 0

    # --- Initialize Data Generator ---
    logger.info("Initializing Data Generator...")
    centered = bool(args.centered)
    train_gen = data_loader(args, logger, split='train', phase='training', centered=centered)
    val_gen = data_loader(args, logger, split='val', phase='testing', centered=centered)
    if centered:
        logger.info("Scene-centered coordinate transformation enabled")
    else:
        logger.info("Scene-centered coordinate transformation disabled")
    
    # Calculate iterations (Approximate)
    iterations_per_epoch = train_gen.num_total_samples // args.batch_size
    if iterations_per_epoch == 0: iterations_per_epoch = 1
    # num_iterations = args.num_epochs * iterations_per_epoch
    
    logger.info(f'Start Training. Epochs: {args.num_epochs}, Batch Size: {args.batch_size}')

    history = {
        'train_losses': defaultdict(list), # Stores G_l2, D_loss, etc. per epoch/iter
        'val_metrics': defaultdict(list)   # Stores ADE, FDE per check
    }

    # 0107 ADD: noise_std
    initial_noise_std = args.noise_std  
    noise_decay_epochs = 20 

    batcher = SmartBatcher(train_gen, args.batch_size, augment=args.augment, max_agents_limit=50, conn_dist=args.conn_dist)
    logger.info(f"Used SmartBatcher to fetch batch, batch size: {args.batch_size}, agent limit: {batcher.effective_limit*2}, conn_dist: {args.conn_dist}")
    
    # Main training loop
    while epoch < args.num_epochs:
        epoch += 1
        # train_gen.shuffle()
        batcher.reset()

        logger.info(f'Starting epoch {epoch}')
        is_warmup = epoch <= args.warmup_epochs
        if is_warmup:
            logger.info(f"WARMUP PHASE: Training Generator Only (Epoch {epoch}/{args.warmup_epochs})")
        
        # Initialize epoch accumulators for losses
        epoch_d_losses = defaultdict(list)
        epoch_g_losses = defaultdict(list)
        batch_count = 0

        # 0107 ADD: noise_std
        current_noise_std = initial_noise_std * max(0.0, 1.0 - (epoch / noise_decay_epochs))
        logger.info(f"Current noise std: {current_noise_std}")

        while batcher.has_data():
        # for itr in range(iterations_per_epoch):
        #     raw_batch = fetch_and_collate_batch(train_gen, args.batch_size, augment=args.augment)
            raw_batch = batcher.next_batch()
            if raw_batch is None: 
                continue
                
            # 2. Convert to Tensors
            batch = prepare_batch(raw_batch, device)
            
            # 3. Optimization
            if not is_warmup:
                for _ in range(args.d_steps):
                    losses_d = discriminator_step(args, batch, generator, discriminator, gan_d_loss, optimizer_d, scaler, device, current_noise_std)
                    losses_d = discriminator_step(args, batch, generator, discriminator, d_hinge_loss, optimizer_d, scaler, device, current_noise_std)
            else:
                losses_d = {'D_loss': 0.0}
            
            for _ in range(args.g_steps):
                # losses_g = generator_step(args, batch, generator, discriminator, gan_g_loss, optimizer_g, scaler, device, is_warmup, current_noise_std)
                losses_g = generator_step(args, batch, generator, discriminator, g_hinge_loss, optimizer_g, scaler, device, is_warmup, current_noise_std)
            
            t += 1
            batch_count += 1

            # Accumulate losses for epoch average
            for k, v in losses_d.items():
                epoch_d_losses[k].append(v)
            for k, v in losses_g.items():
                epoch_g_losses[k].append(v)

            # Console logging (every print_every iterations)
            if t % args.print_every == 0:
                log_str = f'[Ep {epoch}][Iter {t}] '
                log_str += f'D_loss: {losses_d["D_loss"]:.4f} '
                log_str += f'| G_adv: {losses_g["G_adv"]:.4f} '
                log_str += f'| G_l2: {losses_g["G_l2"]:.4f} '
                if args.use_cvae:
                    log_str += f'| G_kl: {losses_g.get("G_kl", 0):.4f}'
                logger.info(log_str)
    
        # --- SAVE EPOCH AVERAGE, MIN, MAX LOSSES TO HISTORY ---
        if batch_count > 0:
            for k in epoch_d_losses:
                loss_values = epoch_d_losses[k]
                avg_loss = np.mean(loss_values)
                min_loss = np.min(loss_values)
                max_loss = np.max(loss_values)
                checkpoint['D_losses'][k].append(avg_loss)
                checkpoint['D_losses'][f'{k}_min'].append(min_loss)
                checkpoint['D_losses'][f'{k}_max'].append(max_loss)
            for k in epoch_g_losses:
                loss_values = epoch_g_losses[k]
                avg_loss = np.mean(loss_values)
                min_loss = np.min(loss_values)
                max_loss = np.max(loss_values)
                checkpoint['G_losses'][k].append(avg_loss)
                checkpoint['G_losses'][f'{k}_min'].append(min_loss)
                checkpoint['G_losses'][f'{k}_max'].append(max_loss)
            checkpoint['losses_ts'].append(epoch)  # Use epoch number instead of iteration
            
            # Log epoch summary
            log_str = f'[Epoch {epoch} Summary] '
            if 'D_loss' in epoch_d_losses:
                log_str += f'Avg D_loss: {np.mean(epoch_d_losses["D_loss"]):.4f} '
            if 'G_adv' in epoch_g_losses:
                log_str += f'| Avg G_adv: {np.mean(epoch_g_losses["G_adv"]):.4f} '
            if 'G_l2' in epoch_g_losses:
                log_str += f'| Avg G_l2: {np.mean(epoch_g_losses["G_l2"]):.4f} '
            if args.use_cvae and 'G_kl' in epoch_g_losses:
                log_str += f'| Avg G_kl: {np.mean(epoch_g_losses["G_kl"]):.4f}'
            logger.info(log_str)

        logger.info('Checking validation...')
        # Use k=1 for fast validation during training (can be increased for more accurate metrics)
        val_metrics = check_accuracy(args, val_gen, generator, limit=True, k=20, augment=False)
        logger.info(f'[Val] ADE: {val_metrics["ade"]:.4f} | FDE: {val_metrics["fde"]:.4f}')
        
        # --- SAVE VAL METRICS TO HISTORY ---
        for k, v in val_metrics.items():
            checkpoint['metrics_val'][k].append(v)
        checkpoint['sample_ts'].append(epoch)  # Use epoch number instead of iteration

        # --- STEP LEARNING RATE SCHEDULERS ---
        # Only step schedulers after at least one optimizer step has been performed
        # This prevents the warning about calling scheduler.step() before optimizer.step()
        if batch_count > 0:
            # Step ReduceLROnPlateau schedulers based on validation metric
            if scheduler_g is not None:
                if args.scheduler_type == 'plateau':
                    scheduler_g.step(val_metrics['ade'])
                elif args.scheduler_type in ['step', 'exponential', 'cosine']:
                    scheduler_g.step()
            
            if scheduler_d is not None:
                if args.scheduler_type == 'plateau':
                    scheduler_d.step(val_metrics['ade'])
                elif args.scheduler_type in ['step', 'exponential', 'cosine']:
                    scheduler_d.step()
        
        # Log current learning rates
        current_lr_g = optimizer_g.param_groups[0]['lr']
        current_lr_d = optimizer_d.param_groups[0]['lr']
        logger.info(f'Current LR - G: {current_lr_g:.2e}, D: {current_lr_d:.2e}')

        # --- CLEAN ARGS (Fix Pickle Error) ---
        saved_args = vars(args).copy()
        if 'get' in saved_args: del saved_args['get']
        
        # Update State
        checkpoint['args'] = saved_args
        checkpoint['counters']['t'] = t
        checkpoint['counters']['epoch'] = epoch
        checkpoint['g_state'] = generator.state_dict()
        checkpoint['g_optim_state'] = optimizer_g.state_dict()
        checkpoint['d_state'] = discriminator.state_dict()
        checkpoint['d_optim_state'] = optimizer_d.state_dict()
        
        # Save scheduler states
        checkpoint['g_scheduler_state'] = scheduler_g.state_dict() if scheduler_g is not None else None
        checkpoint['d_scheduler_state'] = scheduler_d.state_dict() if scheduler_d is not None else None
        # Save Latest
        latest_path = os.path.join(args.output_dir, f'{args.checkpoint_name}_latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save Best (based on ADE + FDE)
        current_ade_fde = val_metrics['ade'] + val_metrics['fde']
        if current_ade_fde < checkpoint['best_ade_fde']:
            checkpoint['best_ade_fde'] = current_ade_fde
            checkpoint['best_ade'] = val_metrics['ade']  # Keep for backward compatibility
            best_path = os.path.join(args.output_dir, f'{args.checkpoint_name}_best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f'New Best Model (ADE+FDE: {current_ade_fde:.4f}, ADE: {val_metrics["ade"]:.4f}, FDE: {val_metrics["fde"]:.4f}) saved: {best_path}')
        
        # Plot training curves at the end of each epoch
        if PLOTTING_AVAILABLE:
            try:
                plot_training_curves(checkpoint, output_dir=args.output_dir, smooth_window=1)
                logger.info('Training curves updated')
            except Exception as e:
                logger.warning(f'Failed to plot training curves: {e}')


def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d, scaler, device, noise_std):
    losses = {}
    optimizer_d.zero_grad()

    # 1. Forward & Loss
    # with torch.amp.autocast('cuda'):
    # with torch.cuda.amp.autocast(enabled=(args.device.type == 'cuda')):
    with torch.cuda.amp.autocast(enabled=False):
        # Generate Fake (Detach to stop gradients to Generator)
        with torch.no_grad():
            pred_fake_abs, _ = generator(batch) 
            pred_fake_abs = pred_fake_abs.permute(1, 0, 2).detach()

        # Get Real Data
        pred_real_abs = batch['fut_motion'] # [Time, Agents, 2]
        
        # Discriminator Forward
        scores_fake = discriminator(batch['pre_motion'], pred_fake_abs, batch['agent_mask'], batch['agent_num'])
        scores_real = discriminator(batch['pre_motion'], pred_real_abs, batch['agent_mask'], batch['agent_num'])

        # Compute Loss
        loss = d_loss_fn(scores_real, scores_fake)
        losses['D_loss'] = loss.item()

    # 3. Backward with Scaler
    scaler.scale(loss).backward()
    
    # 4. Gradient Clipping (must unscale first)
    if args.clipping_threshold_d > 0:
        scaler.unscale_(optimizer_d)
        nn.utils.clip_grad_norm_(discriminator.parameters(), args.clipping_threshold_d)
    
    # 5. Optimizer Step with Scaler
    scaler.step(optimizer_d)
    scaler.update()

    return losses


def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g, scaler, device, is_warmup=False, noise_std=0.0):
    """
    Generator optimization step with optional Best-of-K (Variety Loss).
    
    Args:
        k (int): Number of samples to generate for Variety Loss. 
                 If k=1, standard GAN training.
                 If k>1, minimizes L2 error of the best sample among k generations.
    """
    losses = {}
    k = args.k
    
    optimizer_g.zero_grad()

    with torch.cuda.amp.autocast(enabled=False):
        loss = torch.zeros(1, device=device)
        
        # Ground Truth & Pre-processing
        # 1. Convert GT to [Batch, Time, 2] for unified calculation
        pred_real_norm = batch['fut_motion'].permute(1, 0, 2) 
        
        # 2. Prepare current position for restoration (Residual Connection)
        # [Batch, 2] -> [Batch, 1, 2] (convenient for broadcasting addition to Time dimension)
        # agent_current_pos = batch['agent_current_pos'].unsqueeze(1) 
        
        loss_mask = batch.get('fut_mask', None)
        if loss_mask is not None:
            # [Time, agents] -> 
            loss_mask = loss_mask.transpose(0, 1)   # [agents, Time]

        # --- Forward Generator ---
        if k == 1:
            pred_fake_offset, data_dict = generator(batch) # Output: [agents, time, 2]
            
            # Restore coordinates: [Batch, 1, 2] + [Batch, Time, 2]
            pred_fake_norm = pred_fake_offset  #+  agent_current_pos
            
            best_pred_fake = pred_fake_norm
            best_data_dict = data_dict
        else:
            # Variety Loss Logic (Best-of-K)
            stack_preds, data_dicts_k = generator(batch, k=k)  # Output: [agents, K, time, 2]
            stack_preds = stack_preds.permute(1, 0, 2, 3) # [K, Agents, Time, 2]

            best_pred_fake, best_l2_sum = select_best_k_scene(
                stack_preds, 
                pred_real_norm, 
                batch['seq_start_end'], 
                loss_mask
            )
            
        # Normalize the loss (Equivalent to l2_loss mode='average')
        if loss_mask is not None:
            # Divide by total number of valid time steps in the batch
            l2 = best_l2_sum / torch.sum(loss_mask)
        else:
            # Divide by total elements (Agents * Time)
            l2 = best_l2_sum / (best_pred_fake.shape[0] * best_pred_fake.shape[1])
        loss = loss + args.l2_loss_weight * l2
        losses['G_l2'] = l2.item()

        
        if args.use_cvae:
            q_dist = best_data_dict['q_z_dist']
            p_dist = best_data_dict['p_z_dist_infer']
            kl = torch.distributions.kl.kl_divergence(q_dist, p_dist).sum(dim=-1).mean()
            loss = loss + args.kl_weight * kl
            losses['G_kl'] = kl.item()
        # breakpoint()
        if not is_warmup:
            # Need: [Time, Total_Agents, 2]
            # best_pred_fake = best_pred_fake.permute(1, 0, 2)
            # scores_fake = discriminator(batch['pre_motion'], best_pred_fake, batch['agent_mask'], batch['agent_num'])
            
            # 0109 Modify: Pick last k instead of feeding best traj to D
            random_idx = torch.randint(0, k, (1,)).item()
            pred_fake = stack_preds[random_idx].permute(1, 0, 2)
            scores_fake = discriminator(batch['pre_motion'], pred_fake, batch['agent_mask'], batch['agent_num'])
            
            loss_adv = g_loss_fn(scores_fake)
            loss = loss + loss_adv
            losses['G_adv'] = loss_adv.item()
        else:
            losses['G_adv'] = 0.0

    # Backward & Step with Scaler
    scaler.scale(loss).backward()
    
    if args.clipping_threshold_g > 0:
        scaler.unscale_(optimizer_g)
        nn.utils.clip_grad_norm_(generator.parameters(), args.clipping_threshold_g)
        
    scaler.step(optimizer_g)
    scaler.update()

    return losses


if __name__ == '__main__':
    # 1. Parse Command Line Arguments
    args = parser.parse_args()

    # 2. Load YAML (if provided) and update args
    if args.cfg and os.path.exists(args.cfg):
        with open(args.cfg, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        
        # Update args with values from YAML
        # Note: Command line defaults might override YAML if we aren't careful.
        # Ideally, we set parser defaults to None or handle overrides explicitly.
        # Here, we overwrite args with YAML values.
        for key, value in cfg_dict.items():
            # Only set if the key exists in args to avoid arbitrary injection
            if hasattr(args, key):
                # Convert boolean/string values to appropriate types for certain arguments
                if key == 'use_gpu':
                    # Ensure use_gpu is an integer (0 or 1)
                    if isinstance(value, bool):
                        value = 1 if value else 0
                    elif isinstance(value, str):
                        value = int(value)
                    else:
                        value = int(value)
                elif key == 'gpu_num':
                    # Ensure gpu_num is a string (for device selection)
                    value = str(value)
                setattr(args, key, value)
            else:
                print(f"Warning: Key '{key}' in YAML not found in argparse definitions.")
    args.get = lambda key, default=None: getattr(args, key, default)
    # 3. Run Main
    main(args)
