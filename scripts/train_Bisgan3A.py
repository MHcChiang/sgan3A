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
from model.sgan3A import AgentFormerGenerator, AgentFormerDiscriminator, BiGATLatentEncoder
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
# parser.add_argument('--g_steps', default=1, type=int)
# parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--clipping_threshold_d', default=0, type=float)
parser.add_argument('--warmup_epochs', default=0, type=int, help='Number of epochs to train G only on L2 and Lz loss')
# parser.add_argument('--k', default=1, type=int, help='Number of samples for Variety Loss')

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
# parser.add_argument('--use_cvae', default=0, type=int, help='1 to enable CVAE (KL loss), 0 for Pure GAN')
parser.add_argument('--kl_weight', default=1.0, type=float)
parser.add_argument('--l2_loss_weight', default=1.0, type=float)
parser.add_argument('--lz_weight', default=1.0, type=float)

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
        'out_mlp_dim': [512, 256], # Standard MLP head dims 
        'input_type': [args.input_type]
    }
    
    args.future_encoder = {
        'nlayer': args.enc_layers,
        'out_mlp_dim': [512, 256],
        'input_type': [args.input_type]
    }

    # We pass 'args' as 'cfg' to the models
    generator = AgentFormerGenerator(args).to(device)
    generator.apply(init_weights)
    
    discriminator = AgentFormerDiscriminator(args).to(device)
    discriminator.apply(init_weights)

    # Latent Encoder
    latent_encoder = BiGATLatentEncoder(args.future_encoder, generator.ctx).to(device)
    latent_encoder.apply(init_weights)  

    logger.info("------------- Generator Architecture -------------")
    logger.info(str(generator))
    logger.info("--------------------------------------------------")

    logger.info("----------- Discriminator Architecture -----------")
    logger.info(str(discriminator))
    logger.info("--------------------------------------------------")

    logger.info("----------- Latent Encoder Architecture -----------")
    logger.info(str(latent_encoder))
    logger.info("---------------------------------------------------")

    optimizer_g = optim.Adam(
        list(generator.parameters()) + list(latent_encoder.parameters()),
        lr=args.g_learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)

    # --- Initialize Learning Rate Schedulers ---
    scheduler_g = create_scheduler(optimizer_g, args.scheduler_type, args.num_epochs, args)
    scheduler_d = create_scheduler(optimizer_d, args.scheduler_type, args.num_epochs, args)

    
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
        'e_state': None,
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
        if 'e_state' in checkpoint and checkpoint['e_state'] is not None:
            latent_encoder.load_state_dict(checkpoint['e_state'])
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
    org_kl_weight = args.kl_weight
    org_l2_loss_weight = args.l2_loss_weight
    org_lz_weight = args.lz_weight

    # Main training loop
    while epoch < args.num_epochs:
        epoch += 1
        batcher.reset()

        logger.info(f'Starting epoch {epoch}')
        
        # Initialize epoch accumulators for losses
        epoch_losses = defaultdict(list)
        batch_count = 0

        # 0107 ADD: noise_std
        current_noise_std = initial_noise_std * max(0.0, 1.0 - (epoch / noise_decay_epochs))
        logger.info(f"Current noise std: {current_noise_std}")

        # Test: KL Annealing + l2 boomup
        if epoch <= 10:
            args.kl_weight = org_kl_weight * 0.1
            args.l2_loss_weight = org_l2_loss_weight * 10.0
            args.lz_weight = org_lz_weight * 3.0
        else:
            args.kl_weight = org_kl_weight
            args.l2_loss_weight = org_l2_loss_weight
            args.lz_weight = org_lz_weight
        logger.info(f"Current KL weight: {args.kl_weight}| l2 weight: {args.l2_loss_weight}| lz weight: {args.lz_weight}")

        # warmup phase
        is_warmup = epoch <= args.warmup_epochs
        if is_warmup:
            logger.info(f"WARMUP PHASE: Training Generator Only (Epoch {epoch}/{args.warmup_epochs})")
        
        while batcher.has_data():
            raw_batch = batcher.next_batch()
            if raw_batch is None: continue
                
            # 2. Convert to Tensors
            batch = prepare_batch(raw_batch, device)
            
            # 3. Optimization    
            step_losses = biGAN_step(
                args, batch, generator, discriminator, latent_encoder, 
                d_hinge_loss, g_hinge_loss, # gan_d_loss, gan_g_loss,         #
                optimizer_g, optimizer_d, device, is_warmup)

            t += 1
            batch_count += 1

            # Console logging (every print_every iterations)
            if t % args.print_every == 0:
                log_str = f'[Ep {epoch}][Iter {t}] '
                log_str += f'D_rand: {step_losses["D_rand"]:.4f}'
                log_str += f'| G_rand: {step_losses["G_rand"]:.4f}|'
                log_str += f'| D_rec: {step_losses["D_rec"]:.4f} '
                log_str += f'| G_rec: {step_losses["G_rec"]:.4f} '
                log_str += f'| G_l2: {step_losses["G_l2"]:.4f} '
                log_str += f'| L_z: {step_losses["L_z"]:.4f} '
                log_str += f'| L_kl: {step_losses["L_kl"]:.4f} '
                logger.info(log_str)

            # Collect step losses for log (key->to list of step losses)
            for k, v in step_losses.items():
                epoch_losses[k].append(v)

        # ==========================================
        # End of Epoch: Process & Save Logs
        # ==========================================
        # --- SAVE EPOCH AVERAGE, MIN, MAX LOSSES TO HISTORY ---
        if batch_count > 0:
            log_summary_parts = [f'[Epoch {epoch} Summary]']
            
            for k, v_list in epoch_losses.items():
                avg_val = np.mean(v_list)
                min_val = np.min(v_list)
                max_val = np.max(v_list)

                # Store raw training losses (all keys from biGAN_step)
                checkpoint['metrics_train'][k].append(avg_val)
                checkpoint['metrics_train'][f'{k}_min'].append(min_val)
                checkpoint['metrics_train'][f'{k}_max'].append(max_val)

                # Legacy keys for plotting/compatibility (stage 1 losses)
                if k == 'D_rand':
                    checkpoint['D_losses']['D_loss'].append(avg_val)
                    checkpoint['D_losses']['D_loss_min'].append(min_val)
                    checkpoint['D_losses']['D_loss_max'].append(max_val)
                if k == 'G_rand':
                    checkpoint['G_losses']['G_adv'].append(avg_val)
                    checkpoint['G_losses']['G_adv_min'].append(min_val)
                    checkpoint['G_losses']['G_adv_max'].append(max_val)
                if k == 'G_l2':
                    checkpoint['G_losses']['G_l2'].append(avg_val)
                    checkpoint['G_losses']['G_l2_min'].append(min_val)
                    checkpoint['G_losses']['G_l2_max'].append(max_val)

                # get avg loss for log (all loss items)
                log_summary_parts.append(f'{k}: {avg_val:.4f}')

            checkpoint['losses_ts'].append(epoch)
            log_str = f'[Epoch {epoch} Summary] ' + " | ".join(log_summary_parts[1:])
            logger.info(log_str)

        # --- Validation ---
        logger.info('Checking validation...')
        val_metrics = check_accuracy(args, val_gen, generator, limit=True, k=20, augment=False)
        logger.info(f'[Val] ADE: {val_metrics["ade"]:.4f} | FDE: {val_metrics["fde"]:.4f}')
        
        # --- SAVE VAL METRICS TO HISTORY ---
        for k, v in val_metrics.items():
            checkpoint['metrics_val'][k].append(v)
        checkpoint['sample_ts'].append(epoch)  # Use epoch number instead of iteration

        # --- LEARNING RATE SCHEDULERS ---
        # Only step schedulers after at least one optimizer step has been performed
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
        
        # # Log current learning rates
        # current_lr_g = optimizer_g.param_groups[0]['lr']
        # current_lr_d = optimizer_d.param_groups[0]['lr']
        # logger.info(f'Current LR - G: {current_lr_g:.2e}, D: {current_lr_d:.2e}')

        # # --- CLEAN ARGS (Fix Pickle Error) ---
        # saved_args = vars(args).copy()
        # if 'get' in saved_args: del saved_args['get']
        
        # Update State 
        # checkpoint['args'] = saved_args
        # Save args for checkpoint inspection/compatibility
        saved_args = vars(args).copy()
        if 'get' in saved_args:
            del saved_args['get']
        checkpoint['args'] = saved_args

        checkpoint['counters']['t'] = t
        checkpoint['counters']['epoch'] = epoch
        checkpoint['g_state'] = generator.state_dict()
        checkpoint['g_optim_state'] = optimizer_g.state_dict()
        checkpoint['d_state'] = discriminator.state_dict()
        checkpoint['d_optim_state'] = optimizer_d.state_dict()
        checkpoint['e_state'] = latent_encoder.state_dict()

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


def biGAN_step(args , batch, generator, discriminator,latent_encoder, d_loss_fn, g_loss_fn, optimizer_g, optimizer_d, device, is_warmup):
    losses = {}
    # ==================================================================
    # 0. Make Fake Trajectories
    # ==================================================================  
    # Generate fake trajectory from Random Noise and latent encoder noise
    # From Random Noise
    z_rand = torch.randn(batch['agent_num'], args.nz).to(device)
    # pre-compute context encoding and generate
    generator.context_encoder(batch)
    ctx_enc = batch['context_enc']
    pred_fake_rand, _ = generator(batch, z=z_rand, pre_compute_context=True) # pred_fake_rand: [Agents, time, 2]
    pred_fake_rand = pred_fake_rand.permute(1, 0, 2).contiguous() # [Time, Agents, 2]
    
    # From noise encoded by Latent Encoder
    generator.context_encoder(batch)
    dist_real = latent_encoder(
        traj_in=batch['fut_motion'], 
        context_enc=batch['context_enc'],
        agent_mask=batch['agent_mask'], 
        agent_num=batch['agent_num']
    )
    z_rec = dist_real.rsample()  # rsample to pass gradient
    pred_rec, _ = generator(batch, z=z_rec, pre_compute_context=True)  # pred_rec: [Agents, time, 2]
    pred_rec = pred_rec.permute(1, 0, 2).contiguous() # [Time, Agents, 2]
        
    # ==================================================================
    # 1. Update Discriminator (D)
    # ==================================================================
    if not is_warmup:
        optimizer_d.zero_grad()
        # --- Stage 1 (Stage 1: z_rand -> G -> Fake) ---
        # Real trajectory score
        pred_real = batch['fut_motion']
        score_real = discriminator(batch['pre_motion'], pred_real, batch['agent_mask'], batch['agent_num'])
        score_fake_rand = discriminator(batch['pre_motion'], pred_fake_rand.detach(), batch['agent_mask'], batch['agent_num'])
        
        # Stage 2 (Stage 2: z_rec -> G -> Fake)
        score_fake_rec = discriminator(batch['pre_motion'], pred_rec.detach(), batch['agent_mask'], batch['agent_num'])

        # --- Compute Full D Loss ---
        # BiGAT Discriminator Loss = log(D(real)) + log(1-D(fake_rand)) + log(1-D(fake_rec))
        ld_rand = d_loss_fn(score_real, score_fake_rand)
        ld_rec = d_loss_fn(score_real, score_fake_rec)
        loss_d = ld_rand + ld_rec

        loss_d.backward()
        if args.clipping_threshold_d > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(), args.clipping_threshold_d)
        optimizer_d.step()
    else:
        ld_rand = torch.tensor(0.0).to(device)
        ld_rec = torch.tensor(0.0).to(device)

    # ==================================================================
    # 2. Update Generator (G) and Latent Encoder (E)
    # ==================================================================
    optimizer_g.zero_grad()
    if not is_warmup:
        # --- Stage 1 (Stage 1: z_rand -> G -> Fake) ---
        # Gan loss for G
        score_fake_rand_g = discriminator(batch['pre_motion'], pred_fake_rand, batch['agent_mask'], batch['agent_num'])
        loss_gan1 = g_loss_fn(score_fake_rand_g)
    else:
        loss_gan1 = torch.tensor(0.0).to(device)

    # L_z = |E(Y_hat)-z|
    dist_fake = latent_encoder(
        traj_in=pred_fake_rand,
        context_enc=ctx_enc,
        agent_mask=batch['agent_mask'],
        agent_num=batch['agent_num']
    )
    z_rec_mu = dist_fake.mu
    loss_z = torch.abs(z_rec_mu - z_rand).mean()

    # --- Stage 2 (Stage 2: z_real -> G -> Fake) ---
    # traj reconstruction loss (l2 loss)
    loss_mask = batch.get('fut_mask', None)
    if loss_mask is not None:
        loss_mask = loss_mask.transpose(0, 1)   # [agents, Time]
    l2 = l2_loss(pred_rec, batch['fut_motion'], loss_mask)
    losses['G_l2'] = l2.item()
    if not is_warmup:
        # KL loss for this distribution
        loss_kl = dist_real.kl(p=None).sum(dim=-1).mean()
        
        # G loss (Stage 2)
        score_rec_g = discriminator(batch['pre_motion'], pred_rec, batch['agent_mask'], batch['agent_num']) # stage 2
        loss_gan2 = g_loss_fn(score_rec_g)
    else:
        loss_kl = torch.tensor(0.0).to(device)
        loss_gan2 = torch.tensor(0.0).to(device)
 
    # Total loss
    loss = (
        loss_gan1
        + args.lz_weight * loss_z
        + loss_gan2
        + args.kl_weight * loss_kl
        + args.l2_loss_weight * l2
    )

    # Add to log
    losses['G_rand'] = loss_gan1.item()
    losses['D_rand'] = ld_rand.item()
    losses['L_z'] = loss_z.item()
    losses['G_rec'] = loss_gan2.item()
    losses['D_rec'] = ld_rec.item()
    losses['G_l2'] = l2.item()
    losses['L_kl'] = loss_kl.item()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(generator.parameters(), args.clipping_threshold_g)
    optimizer_g.step()

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
