import argparse
import gc
import logging
import os
import sys
import time

from collections import defaultdict

# Add project root to Python path to enable imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from model.data.loader import data_loader
from model.losses import (
    gan_g_loss,
    gan_d_loss,
    l2_loss,
    displacement_error,
    final_displacement_error,
)
from model.utils import int_tuple, bool_flag, get_total_norm
from model.utils import relative_to_abs, get_dset_path

from model.sgan3A import AgentFormerGenerator, AgentFormerDiscriminator

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='zara1', type=str,
                    help='Dataset name: zara1, zara2, eth, hotel, univ, or "all" to use all datasets')
parser.add_argument('--delim', default=' ')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=200, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=1024, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=128, type=int)
parser.add_argument('--noise_dim', default=None, type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=5e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=1024, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=5e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=0, type=float)
parser.add_argument('--best_k', default=1, type=int)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Model Architecture
parser.add_argument('--model_type', default='lstm', type=str,
                    choices=['lstm', 'transformer'],
                    help='Model architecture: lstm or transformer')

# Configuration file
parser.add_argument('--config', type=str, default=None,
                    help='Path to YAML configuration file. If provided, arguments from YAML will override defaults.')

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
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
    """
    Get device object based on args.use_gpu and CUDA availability.
    Device-agnostic: works on CPU (Mac) and GPU (cloud).
    
    Returns:
        torch.device object
    """
    if args.use_gpu == 1 and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def build_generator_cfg(args):
    return {
        'obs_len': args.obs_len,
        'pred_len': args.pred_len,
        'embedding_dim': args.embedding_dim,
        'encoder_h_dim': args.encoder_h_dim_g,
        'decoder_h_dim': args.decoder_h_dim_g,
        'mlp_dim': args.mlp_dim,
        'num_layers': args.num_layers,
        'noise_dim': args.noise_dim,
        'noise_type': args.noise_type,
        'noise_mix_type': args.noise_mix_type,
        'pooling_type': args.pooling_type,
        'pool_every_timestep': args.pool_every_timestep,
        'dropout': args.dropout,
        'bottleneck_dim': args.bottleneck_dim,
        'activation': 'relu',
        'batch_norm': args.batch_norm,
        'neighborhood_size': args.neighborhood_size,
        'grid_size': args.grid_size,
    }


def build_discriminator_cfg(args):
    return {
        'obs_len': args.obs_len,
        'pred_len': args.pred_len,
        'embedding_dim': args.embedding_dim,
        'h_dim': args.encoder_h_dim_d,
        'mlp_dim': args.mlp_dim,
        'num_layers': args.num_layers,
        'activation': 'relu',
        'batch_norm': args.batch_norm,
        'dropout': args.dropout,
        'd_type': args.d_type,
    }


def main(args):
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f'Created output directory: {args.output_dir}')
    else:
        logger.info(f'Using existing output directory: {args.output_dir}')
    
    # Only set CUDA_VISIBLE_DEVICES if using GPU and CUDA is available
    if args.use_gpu == 1 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    
    # New: Handle "all" datasets option
    if args.dataset_name.lower() == 'all':
        # List of all available datasets
        all_datasets = ['zara1', 'zara2', 'eth', 'hotel', 'univ']
        logger.info(f'Loading all datasets: {all_datasets}')
        train_paths = [get_dset_path(dset, 'train') for dset in all_datasets]
        val_paths = [get_dset_path(dset, 'val') for dset in all_datasets]
        
        # Verify all paths exist
        for dset, train_path in zip(all_datasets, train_paths):
            if not os.path.exists(train_path):
                logger.warning(f'Dataset {dset} train path not found: {train_path}')
    else:
        # Single dataset
        train_paths = get_dset_path(args.dataset_name, 'train')
        val_paths = get_dset_path(args.dataset_name, 'val')
        logger.info(f'Loading single dataset: {args.dataset_name}')

    # Get device object (device-agnostic for Mac/CPU compatibility)
    device = get_device(args)
    logger.info(f'Using device: {device}')

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_paths)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_paths)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    # Choose model architecture
    if args.model_type != 'transformer':
        raise ValueError('Only the AgentFormer (transformer) generator is supported in this build.')
    logger.info('Using AgentFormer generator/discriminator')
    generator_cfg = build_generator_cfg(args)
    discriminator_cfg = build_discriminator_cfg(args)
    generator = TrajectoryGenerator_AgentFormer.from_cfg(generator_cfg)
    discriminator = TrajectoryDiscriminator_AgentFormer.from_cfg(discriminator_cfg)

    generator.apply(init_weights)
    generator = generator.to(device)
    generator.train()
    logger.info('Here is the generator:')
    logger.info(generator)

    discriminator.apply(init_weights)
    discriminator = discriminator.to(device)
    discriminator.train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.d_learning_rate
    )

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        # Load checkpoint with device mapping (works on CPU/Mac and GPU)
        checkpoint = torch.load(restore_path, map_location=device)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }
    t0 = None
    # Main training loop
    while t < args.num_iterations:
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, generator,
                                              discriminator, d_loss_fn,
                                              optimizer_d, device)
                checkpoint['norm_d'].append(
                    get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                step_type = 'g'
                losses_g = generator_step(args, batch, generator,
                                          discriminator, g_loss_fn,
                                          optimizer_g, device)
                checkpoint['norm_g'].append(
                    get_total_norm(generator.parameters())
                )
                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                for k, v in sorted(losses_d.items()):
                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)
                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    args, val_loader, generator, discriminator, d_loss_fn, device
                )
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    args, train_loader, generator, discriminator,
                    d_loss_fn, device, limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator.state_dict()

                # Save another checkpoint with model weights and optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_with_model.pt' % args.checkpoint_name
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_no_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                key_blacklist = [
                    'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                    'g_optim_state', 'd_optim_state', 'd_best_state',
                    'd_best_nl_state'
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            if t >= args.num_iterations:
                break


def discriminator_step(
    args, batch, generator, discriminator, d_loss_fn, optimizer_d, device
):
    """
    Discriminator training step using AgentFormer's data format directly.
    """
    # batch is in AgentFormer's format (dictionary)
    import numpy as np
    losses = {}
    loss = torch.zeros(1, device=device)

    # Generate fake trajectories
    pred_traj_fake_rel = generator(batch)  # (pred_len, num_agents, 2)
    
    # Convert to absolute for creating discriminator input
    pre_motion_list = batch['pre_motion_3D']
    num_agents = len(pre_motion_list)
    obs_traj = torch.stack([torch.from_numpy(m).float() for m in pre_motion_list], dim=0)
    obs_traj = obs_traj.transpose(0, 1).to(device)  # (obs_len, num_agents, 2)
    
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
    
    # Create data dictionaries for discriminator (AgentFormer format)
    # Real trajectory
    traj_real_data = {
        'pre_motion_3D': batch['pre_motion_3D'],
        'fut_motion_3D': batch['fut_motion_3D'],
        'seq_start_end': batch.get('seq_start_end', [(0, num_agents)])
    }
    
    # Fake trajectory: convert fake predictions back to list format
    traj_fake_data = {
        'pre_motion_3D': batch['pre_motion_3D'],
        'fut_motion_3D': [pred_traj_fake[:, i].cpu().numpy() for i in range(num_agents)],
        'seq_start_end': batch.get('seq_start_end', [(0, num_agents)])
    }

    scores_fake = discriminator(traj_fake_data)
    scores_real = discriminator(traj_real_data)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(
    args, batch, generator, discriminator, g_loss_fn, optimizer_g, device
):
    """
    Generator training step using AgentFormer's data format directly.
    """
    # batch is in AgentFormer's format (dictionary)
    import numpy as np
    losses = {}
    loss = torch.zeros(1, device=device)
    g_l2_loss_rel = []

    # Convert to tensors for loss computation
    pre_motion_list = batch['pre_motion_3D']
    fut_motion_list = batch['fut_motion_3D']
    fut_mask_list = batch['fut_motion_mask']
    num_agents = len(pre_motion_list)
    
    obs_traj = torch.stack([torch.from_numpy(m).float() for m in pre_motion_list], dim=0)
    obs_traj = obs_traj.transpose(0, 1).to(device)  # (obs_len, num_agents, 2)
    
    pred_traj_gt = torch.stack([torch.from_numpy(m).float() for m in fut_motion_list], dim=0)
    pred_traj_gt = pred_traj_gt.transpose(0, 1).to(device)  # (pred_len, num_agents, 2)
    
    # Compute relative trajectories for loss
    pred_traj_gt_rel = pred_traj_gt - torch.cat([obs_traj[-1:], pred_traj_gt[:-1]], dim=0)
    
    # Loss mask
    loss_mask = torch.stack([torch.from_numpy(m).float() for m in fut_mask_list], dim=0)
    loss_mask = loss_mask.transpose(0, 1).to(device)  # (num_agents, pred_len)
    loss_mask = loss_mask[:, args.obs_len:]  # Only future timesteps
    
    seq_start_end = batch.get('seq_start_end', [(0, num_agents)])
    if isinstance(seq_start_end, list):
        seq_start_end = torch.tensor(seq_start_end, dtype=torch.long, device=device)

    # Generate k predictions (variety loss)
    for _ in range(args.best_k):
        pred_traj_fake_rel = generator(batch)  # (pred_len, num_agents, 2)
        
        # Compute L2 loss for each prediction
        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))
                
    # Take minimum over k predictions (variety loss)
    g_l2_loss_sum_rel = torch.zeros(1, device=device)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.tolist():
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    # Get last prediction for discriminator
    pred_traj_fake_rel = generator(batch)
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
    
    # Create discriminator input (AgentFormer format)
    traj_fake_data = {
        'pre_motion_3D': batch['pre_motion_3D'],
        'fut_motion_3D': [pred_traj_fake[:, i].cpu().numpy() for i in range(num_agents)],
        'seq_start_end': batch.get('seq_start_end', [(0, num_agents)])
    }

    scores_fake = discriminator(traj_fake_data)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return losses


def check_accuracy(
    args, loader, generator, discriminator, d_loss_fn, device, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            # batch is in AgentFormer's format (dictionary)
            import numpy as np
            
            # Convert to tensors for metric computation
            pre_motion_list = batch['pre_motion_3D']
            fut_motion_list = batch['fut_motion_3D']
            fut_mask_list = batch['fut_motion_mask']
            num_agents = len(pre_motion_list)
            
            obs_traj = torch.stack([torch.from_numpy(m).float() for m in pre_motion_list], dim=0)
            obs_traj = obs_traj.transpose(0, 1).to(device)
            
            pred_traj_gt = torch.stack([torch.from_numpy(m).float() for m in fut_motion_list], dim=0)
            pred_traj_gt = pred_traj_gt.transpose(0, 1).to(device)
            
            # Compute relative for loss
            pred_traj_gt_rel = pred_traj_gt - torch.cat([obs_traj[-1:], pred_traj_gt[:-1]], dim=0)
            
            loss_mask = torch.stack([torch.from_numpy(m).float() for m in fut_mask_list], dim=0)
            loss_mask = loss_mask.transpose(0, 1).to(device)
            loss_mask = loss_mask[:, args.obs_len:]
            
            # Non-linear ped (simplified)
            non_linear_ped = torch.ones(num_agents, dtype=torch.bool, device=device)
            linear_ped = 1 - non_linear_ped.float()
            
            seq_start_end = batch.get('seq_start_end', [(0, num_agents)])
            if isinstance(seq_start_end, list):
                seq_start_end = torch.tensor(seq_start_end, dtype=torch.long, device=device)

            pred_traj_fake_rel = generator(batch)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            # Create discriminator inputs in AgentFormer format
            traj_real_data = {
                'pre_motion_3D': batch['pre_motion_3D'],
                'fut_motion_3D': batch['fut_motion_3D'],
                'seq_start_end': batch.get('seq_start_end', [(0, num_agents)])
            }
            
            traj_fake_data = {
                'pre_motion_3D': batch['pre_motion_3D'],
                'fut_motion_3D': [pred_traj_fake[:, i].cpu().numpy() for i in range(num_agents)],
                'seq_start_end': batch.get('seq_start_end', [(0, num_agents)])
            }

            scores_fake = discriminator(traj_fake_data)
            scores_real = discriminator(traj_real_data)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            # Remove .data attribute usage (deprecated in modern PyTorch)
            loss_mask_sum += torch.numel(loss_mask)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
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
    # First, parse only the config argument
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--config', type=str, default=None)
    temp_args, remaining_args = temp_parser.parse_known_args()
    
    # Load YAML config if provided
    yaml_config = {}
    if temp_args.config is not None:
        if not os.path.exists(temp_args.config):
            logger.error(f'Config file not found: {temp_args.config}')
            sys.exit(1)
        logger.info(f'Loading configuration from: {temp_args.config}')
        yaml_config = load_config_from_yaml(temp_args.config)
    
    # Convert YAML keys to argparse format (replace '-' with '_', remove '--' prefix)
    yaml_args = []
    for key, value in yaml_config.items():
        key = key.lstrip('--').replace('-', '_')
        if value is not None:
            # Convert value to string for argparse
            if isinstance(value, bool):
                yaml_args.extend([f'--{key}', '1' if value else '0'])
            elif isinstance(value, list):
                # Handle lists (e.g., noise_dim)
                yaml_args.extend([f'--{key}'] + [str(v) for v in value])
            else:
                yaml_args.extend([f'--{key}', str(value)])
    
    # Combine YAML args with command-line args (command-line overrides YAML)
    all_args = yaml_args + remaining_args
    
    # Parse all arguments
    args = parser.parse_args(all_args)
    
    main(args)
