import argparse
import gc
import logging
import os
import sys
import time
import numpy as np
from collections import defaultdict
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append(os.getcwd())
# Import Data Generator (The Iterator based one)
from model.data.dataloader import data_generator
from model.sgan3A import AgentFormerGenerator, AgentFormerDiscriminator
from model.losses import gan_g_loss, gan_d_loss, l2_loss
from utils.logger import Logger

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



def init_weights(m):
    """
    Initialize weights for neural network layers.
    
    This function is applied to all modules in the model using .apply().
    It initializes Linear layers with Kaiming normal initialization, which
    is particularly good for ReLU activations and helps with training stability.
    
    Args:
        m: A PyTorch module (layer) from the model
        
    Usage:
        model.apply(init_weights)  # Applies to all modules recursively
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_device(args):
    return torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')


def create_scheduler(optimizer, scheduler_type, num_epochs, args):
    """
    Create a learning rate scheduler for the given optimizer.
    
    Args:
        optimizer: The optimizer to attach the scheduler to
        scheduler_type: Type of scheduler ('none', 'step', 'exponential', 'plateau', 'cosine')
        num_epochs: Total number of epochs for training
        args: Arguments object containing scheduler parameters
    
    Returns:
        Scheduler object or None
    """
    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'step':
        # StepLR: decay every step_size epochs by gamma factor
        step_size = args.scheduler_step_size if args.scheduler_step_size is not None else max(1, int(num_epochs * 0.3))
        gamma = args.scheduler_gamma if args.scheduler_gamma is not None else 0.1
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'exponential':
        # ExponentialLR: decay by gamma factor every epoch
        gamma = args.scheduler_gamma if args.scheduler_gamma is not None else 0.95
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == 'plateau':
        # ReduceLROnPlateau: reduce LR when metric plateaus
        patience = args.scheduler_patience
        factor = args.scheduler_factor
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, verbose=True
        )
    elif scheduler_type == 'cosine':
        # CosineAnnealingLR: cosine annealing over total epochs
        T_max = num_epochs
        eta_min = args.scheduler_min_lr
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ==========================================
# Data Preparation Helper 
# ==========================================
def collate_scenes(scenes, mask=False):
    """
    Merges a list of scene dictionaries into a single batch dictionary.
    Creates Block-Diagonal Mask.
    """
    batch_data = {}
    keys_to_merge = ['pre_motion_3D', 'fut_motion_3D', 'pre_motion_mask', 'fut_motion_mask', 'heading']
    
    for key in keys_to_merge:
        if scenes[0].get(key) is not None:
            merged_list = []
            for s in scenes:
                merged_list.extend(s[key])
            batch_data[key] = merged_list
        else:
            batch_data[key] = None

    agents_per_scene = [len(s['pre_motion_3D']) for s in scenes]
    total_agents = sum(agents_per_scene)
    batch_data['agent_num'] = total_agents

    # Block-Diagonal Mask (-inf = Disconnected)
    if mask:
        big_mask = np.full((total_agents, total_agents), float('-inf'), dtype=np.float32)
        current_idx = 0
        for n_agents in agents_per_scene:
            scene_mask = np.zeros((n_agents, n_agents), dtype=np.float32)
            big_mask[current_idx : current_idx+n_agents, 
                    current_idx : current_idx+n_agents] = scene_mask
            current_idx += n_agents
            
        batch_data['agent_mask'] = big_mask
    return batch_data


def fetch_and_collate_batch(generator, batch_size):
    """
    Fetches 'batch_size' scenes and collates them.
    """
    scene_samples = []
    while len(scene_samples) < batch_size:
        if not generator.is_epoch_end():
            scene = generator()
            if scene is not None:
                scene_samples.append(scene)
        else:   
            break
            
    if len(scene_samples) > 0:
        return collate_scenes(scene_samples, mask=batch_size>1)
    else:
        return None


def prepare_batch(batch, device, augment=False):
    """
    Converts list of numpy arrays from data_generator into Tensor dictionary.
    Handles 'One Scene' logic correctly.
    Applies Random Rotation Augmentation if enabled.
    """
    def to_tensor(x):
        if isinstance(x, torch.Tensor): return x.clone().detach().float()
        elif isinstance(x, np.ndarray): return torch.from_numpy(x).float()
        else: return torch.tensor(x).float()

    data = {}
    # Stack list -> [Time, Agents, 2]XX -> [Agents, Time, 2]
    pre_motion = torch.stack([to_tensor(m) for m in batch['pre_motion_3D']], dim=0).to(device)
    fut_motion = torch.stack([to_tensor(m) for m in batch['fut_motion_3D']], dim=0).to(device)
    # print(f"load: {pre_motion.shape}")
    # Tranpose (NEED?)
    pre_motion = pre_motion.transpose(0, 1).contiguous() # [Batch, Time, 2]X 
    fut_motion = fut_motion.transpose(0, 1).contiguous()
    # print(f"transpose: {pre_motion.shape}")

    # --- DATA AUGMENTATION (Random Rotation) ---
    if augment:
        # Generate random angle: 0 to 2*pi
        theta = torch.rand(1).to(device) * 2 * np.pi
        c, s = torch.cos(theta), torch.sin(theta)
        
        # Apply to Past and Future
        pre_motion = rotate_tensor(pre_motion,c ,s)
        fut_motion = rotate_tensor(fut_motion, c, s)
        
        # Note: If you use 'heading' (yaw), you must rotate that too
        if 'heading' in batch and batch['heading'] is not None:
             # This part might need adjustment depending on how 'heading' is stored
             # Usually it is a list of floats (radians)
             # batch['heading'] is a list, so we handle it later or ignore if not used
             pass

    data['pre_motion'] = pre_motion
    data['fut_motion'] = fut_motion
    data['agent_num'] = pre_motion.shape[1]
    
    # ... (Rest of the function remains the same) ...
    
    # Create connectivity mask (Using 'pre_motion_mask' if available, else fully connected)
    if 'agent_mask' in batch:
        data['agent_mask'] = to_tensor(batch['agent_mask']).to(device)
    else:
        data['agent_mask'] = torch.zeros(data['agent_num'], data['agent_num']).to(device)

    data['heading'] = torch.zeros(data['agent_num']).to(device)
    data['heading_vec'] = torch.zeros(data['agent_num'], 2).to(device)
    pre_vel = torch.zeros_like(pre_motion)
    # pre_vel[1:] = pre_motion[1:] - pre_motion[:-1]
    data['pre_vel'] = pre_vel
    data['pre_motion_scene_norm'] = pre_motion
    data['agent_enc_shuffle'] = None
    
    if 'fut_motion_mask' in batch:
        mask = torch.stack([to_tensor(m) for m in batch['fut_motion_mask']], dim=0).to(device)
        data['fut_mask'] = mask.transpose(0, 1).contiguous()
    
    return data


# Helper to rotate a tensor [..., 2]
def rotate_tensor(t, c, s):
    # t shape: [..., 2]
    x = t[..., 0]
    y = t[..., 1]
    x_new = x * c - y * s
    y_new = x * s + y * c
    return torch.stack([x_new, y_new], dim=-1)


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

    # --- Initialize Models ---
    args.context_encoder = {
        'nlayer': args.enc_layers,
        'input_type': [args.input_type] # e.g. ['pos']
    }
    
    args.future_decoder = {
        'nlayer': args.dec_layers,
        'out_mlp_dim': [512, 256], # Standard MLP head dims
        'input_type': [args.input_type]
    }
    
    # Also needed if using CVAE mode
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
        'best_ade': float('inf'),
    }

    # --- Restore Checkpoint ---
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir, f'{args.checkpoint_name}_latest.pt')

    
    if restore_path is not None and os.path.isfile(restore_path):
        logger.info(f'Restoring from checkpoint {restore_path}')
        checkpoint = torch.load(restore_path, map_location=device)
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
        best_ade = checkpoint.get('best_ade', float('inf'))
    else:
        logger.info('Starting new training')
        t = 0
        epoch = 0
        best_ade = float('inf')
    
    # --- Initialize Data Generator ---
    logger.info("Initializing Data Generator...")
    train_gen = data_generator(args, logger, split='train', phase='training')
    val_gen = data_generator(args, logger, split='val', phase='testing')
    
    # Calculate iterations (Approximate)
    iterations_per_epoch = train_gen.num_total_samples // args.batch_size
    if iterations_per_epoch == 0: iterations_per_epoch = 1
    num_iterations = args.num_epochs * iterations_per_epoch
    
    logger.info(f'Start Training. Epochs: {args.num_epochs}, Batch Size: {args.batch_size}')

    history = {
        'train_losses': defaultdict(list), # Stores G_l2, D_loss, etc. per epoch/iter
        'val_metrics': defaultdict(list)   # Stores ADE, FDE per check
    }

    # Main training loop
    while t < num_iterations:
        epoch += 1
        train_gen.shuffle()
        logger.info(f'Starting epoch {epoch}')
        
        # Initialize epoch accumulators for losses
        epoch_d_losses = defaultdict(list)
        epoch_g_losses = defaultdict(list)
        batch_count = 0
        
        for itr in range(iterations_per_epoch):
            # 1. Fetch & Collate
            raw_batch = fetch_and_collate_batch(train_gen, args.batch_size)
            
            if raw_batch is None: 
                continue
                
            # 2. Convert to Tensors
            batch = prepare_batch(raw_batch, device, augment=args.augment)
            
            # 3. Optimization
            for _ in range(args.d_steps):
                losses_d = discriminator_step(args, batch, generator, discriminator, gan_d_loss, optimizer_d, device)
            
            for _ in range(args.g_steps):
                losses_g = generator_step(args, batch, generator, discriminator, gan_g_loss, optimizer_g, device)
            
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

            if t >= num_iterations:
                break
        
        # --- SAVE EPOCH AVERAGE LOSSES TO HISTORY ---
        if batch_count > 0:
            for k in epoch_d_losses:
                avg_loss = np.mean(epoch_d_losses[k])
                checkpoint['D_losses'][k].append(avg_loss)
            for k in epoch_g_losses:
                avg_loss = np.mean(epoch_g_losses[k])
                checkpoint['G_losses'][k].append(avg_loss)
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
        val_metrics = check_accuracy(args, val_gen, generator, limit=True)
        logger.info(f'[Val] ADE: {val_metrics["ade"]:.4f} | FDE: {val_metrics["fde"]:.4f}')
        
        # --- SAVE VAL METRICS TO HISTORY ---
        for k, v in val_metrics.items():
            checkpoint['metrics_val'][k].append(v)
        checkpoint['sample_ts'].append(epoch)  # Use epoch number instead of iteration

        # --- STEP LEARNING RATE SCHEDULERS ---
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
        
        # Save Best
        if val_metrics['ade'] < checkpoint['best_ade']:
            checkpoint['best_ade'] = val_metrics['ade']
            best_path = os.path.join(args.output_dir, f'{args.checkpoint_name}_best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f'New Best Model (ADE {checkpoint["best_ade"]:.4f}) saved: {best_path}')


def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d, device):
    losses = {}
    loss = torch.zeros(1, device=device)

    # 1. Generate Fake (Detach to stop gradients to Generator)
    # Note: Generator forward returns (pred_fake, data_dict)
    with torch.no_grad():
        pred_fake_abs, _ = generator(batch) 
        # pred_fake_abs = pred_fake_abs.detach()
        pred_fake_abs = pred_fake_abs.permute(1, 0, 2).detach()

    # 2. Get Real Data (Absolute)
    pred_real_abs = batch['fut_motion'] # [Time, Agents, 2]
    
    # 3. Discriminator Forward
    # Inputs are [Time, Agents, 2]
    scores_fake = discriminator(batch['pre_motion'], pred_fake_abs, batch['agent_mask'], batch['agent_num'])
    scores_real = discriminator(batch['pre_motion'], pred_real_abs, batch['agent_mask'], batch['agent_num'])

    # 4. Compute Loss
    data_loss = d_loss_fn(scores_real, scores_fake)
    loss += data_loss
    losses['D_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(), args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g, device):
    losses = {}
    loss = torch.zeros(1, device=device)
    
    # 1. Forward Generator
    # If CVAE is enabled, this encodes future -> z. Else samples random z.
    pred_fake_abs, data_dict = generator(batch)
    pred_real_abs = batch['fut_motion']
    pred_fake_abs = pred_fake_abs.permute(1, 0, 2)

    # 2. Reconstruction Loss (L2 on Absolute Coords)
    loss_mask = batch.get('fut_mask', None)
    if loss_mask is not None:
        loss_mask = loss_mask.transpose(0, 1)
            
    # l2 = l2_loss(pred_fake_abs, pred_real_abs, loss_mask, mode='raw')
    l2 = l2_loss(pred_fake_abs, pred_real_abs, loss_mask, mode='average')
    loss += args.l2_loss_weight * l2
    losses['G_l2'] = l2.item()

    # 3. KL Divergence (If CVAE mode is ON)
    if args.use_cvae:
        q_dist = data_dict['q_z_dist']
        p_dist = data_dict['p_z_dist_infer'] # Prior (usually N(0,1))
        # Compute KL(Posterior || Prior)
        kl = torch.distributions.kl.kl_divergence(q_dist, p_dist).sum(dim=-1).mean()
        loss += args.kl_weight * kl
        losses['G_kl'] = kl.item()

    # 4. Adversarial Loss
    scores_fake = discriminator(batch['pre_motion'], pred_fake_abs, batch['agent_mask'], batch['agent_num'])
    loss_adv = g_loss_fn(scores_fake)
    loss += loss_adv
    losses['G_adv'] = loss_adv.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(generator.parameters(), args.clipping_threshold_g)
    optimizer_g.step()

    return losses


def check_accuracy(args, loader, generator, limit=False):
    """
    Evaluates the generator on the validation set.
    Calculates ADE (Average Displacement Error), FDE (Final Displacement Error), and L2 loss.
    """
    metrics = {}
    ade_outer, fde_outer = [], []
    l2_losses = []
    total_traj = 0
    
    generator.eval()
    # loader.reset() # Ensure we start from the beginning
    
    with torch.no_grad():
        while not loader.is_epoch_end():
            # 1. Fetch & Collate (Reuse the same batching logic)
            raw_batch = fetch_and_collate_batch(loader, args.batch_size)
            if raw_batch is None: continue
            
            # 2. Convert to Tensors
            batch = prepare_batch(raw_batch, get_device(args))
            
            # 3. Forward Pass (Inference Mode)
            # Use same z sampling as training or fixed z for determinism? 
            # Usually random z for variety metrics, or mean z. 
            # Here we follow standard AgentFormer evaluation (20 samples) or just 1 for quick check.
            # Let's do 1 sample for fast validation during training.
            pred_fake_abs, _ = generator(batch) 
            pred_fake_abs = pred_fake_abs.permute(1, 0, 2)
            pred_real_abs = batch['fut_motion'] # [Time, Agents, 2]
            
            # 4. Calculate Metrics (ADE / FDE)
            diff = pred_fake_abs - pred_real_abs
            dist = torch.norm(diff, dim=-1) # [12, N]
            
            ade = dist.mean(dim=0) # Mean over time -> [N]
            fde = dist[-1]         # Last timestep -> [N]
            
            # Filter valid (if using masks)
            if 'fut_mask' in batch:
                # Mask is already [Time, Agents] from prepare_batch
                valid_mask = batch['fut_mask'] > 0
                
                # ADE: Mean over Time (masked)
                # Sum error over time / Sum valid frames over time
                ade = (dist * valid_mask).sum(dim=0) / (valid_mask.sum(dim=0) + 1e-6)
                
                # FDE: Last Valid Frame Error
                # We assume the last frame is the target frame.
                fde = dist[-1] * valid_mask[-1]
            else:
                ade = dist.mean(dim=0)
                fde = dist[-1]
            
            # 5. Calculate L2 Loss (same as in generator_step)
            loss_mask = batch.get('fut_mask', None)
            if loss_mask is not None:
                loss_mask = loss_mask.transpose(0, 1)
            l2 = l2_loss(pred_fake_abs, pred_real_abs, loss_mask, mode='average')
            l2_losses.append(l2.item())
            
            ade_outer.append(ade)
            fde_outer.append(fde)
            total_traj += batch['agent_num']
            
            if limit and total_traj >= args.num_samples_check:
                break

    # Aggregate results
    ade_all = torch.cat(ade_outer).mean().item()
    fde_all = torch.cat(fde_outer).mean().item()
    l2_all = np.mean(l2_losses) if l2_losses else 0.0
    
    metrics['ade'] = ade_all
    metrics['fde'] = fde_all
    metrics['l2'] = l2_all
    
    generator.train() # Switch back to train mode
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


def load_config_from_yaml(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary of configuration values
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config if config else {}


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
                setattr(args, key, value)
            else:
                print(f"Warning: Key '{key}' in YAML not found in argparse definitions.")
    args.get = lambda key, default=None: getattr(args, key, default)
    # 3. Run Main
    main(args)
