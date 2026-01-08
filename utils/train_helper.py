import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from model.losses import (
    l2_loss,
    select_best_k_scene,
    displacement_error,
    final_displacement_error,
)




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
    Get the appropriate device (CPU or GPU) based on arguments.
    
    Args:
        args: Arguments object with use_gpu and gpu_num attributes
    
    Returns:
        torch.device object
    """
    # Convert use_gpu to int if it's a boolean or string
    use_gpu = args.use_gpu
    if isinstance(use_gpu, bool):
        use_gpu = 1 if use_gpu else 0
    elif isinstance(use_gpu, str):
        use_gpu = int(use_gpu)
    else:
        use_gpu = int(use_gpu)
    
    if use_gpu and torch.cuda.is_available():
        # Select specific GPU if gpu_num is provided
        gpu_num = args.gpu_num
        if isinstance(gpu_num, str):
            gpu_num = int(gpu_num)
        else:
            gpu_num = int(gpu_num)
        
        # Set the default CUDA device
        torch.cuda.set_device(gpu_num)
        device = torch.device(f'cuda:{gpu_num}')
        
        # Verify device is accessible
        try:
            torch.cuda.get_device_properties(gpu_num)
            return device
        except Exception as e:
            print(f"Warning: Could not access GPU {gpu_num}: {e}")
            print("Falling back to CPU")
            return torch.device('cpu')
    else:
        if use_gpu:
            print("Warning: use_gpu is set to 1 but CUDA is not available. Using CPU.")
        return torch.device('cpu')


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
def collate_scenes(scenes, mask=False, conn_dist=10000.0):
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

    # Prepare seq_start_end for variety loss
    seq_start_end = []
    current_idx = 0
    for n_agents in agents_per_scene:
        seq_start_end.append((current_idx, current_idx + n_agents))
        current_idx += n_agents
    batch_data['seq_start_end'] = seq_start_end

    # Block-Diagonal Mask (-inf = Disconnected)
    if mask:
        big_mask = np.full((total_agents, total_agents), float('-inf'), dtype=np.float32) # initialize as -inf
        current_idx = 0
        # 0106 ADD: connectivity mask
        for i, s in enumerate(scenes):
            n_agents = agents_per_scene[i]
            scene_mask = np.zeros((n_agents, n_agents), dtype=np.float32)

            # connectivity mask (Block agent that is too far away)
            if conn_dist < 10000.0: # Do this only if the threshold is valid
                curr_pos = np.stack([agent_traj[-1] for agent_traj in s['pre_motion_3D']], axis=0) # Shape: [n_agents, 2]
                diff = curr_pos[:, None, :] - curr_pos[None, :, :] 
                dist_mat = np.linalg.norm(diff, axis=-1) # Shape: [n_agents, n_agents]
                scene_mask[dist_mat > conn_dist] = float('-inf')

            # Fill connectivity mask into scene mask
            big_mask[current_idx : current_idx+n_agents, 
                    current_idx : current_idx+n_agents] = scene_mask
            current_idx += n_agents
            
        batch_data['agent_mask'] = big_mask
    return batch_data


class SmartBatcher:
    def __init__(self, generator, batch_size, augment=False, max_agents_limit=50, conn_dist=100000.0):
        self.generator = generator
        self.augment = augment
        self.buffer = deque()
        self.conn_dist = conn_dist
        
        # Q1 optimization: Handle augmentation coefficients during initialization
        if self.augment:
            self.target_count = max(1, batch_size // 2)
            self.effective_limit = max_agents_limit // 2
        else:
            self.target_count = batch_size
            self.effective_limit = max_agents_limit

        self.agent_level = int(0.75 * self.effective_limit)
        self.generator_exhausted = False
            
    def reset(self):
        """Called before each epoch starts"""
        self.buffer.clear()
        self.generator.shuffle()
        self.generator_exhausted = False 
        
    def has_data(self):
        """
        Check if there is still data available.
        Key: Never call generator.is_epoch_end() here, only check Buffer and internal Flag.
        """
        if len(self.buffer) > 0:
            return True
        
        return not self.generator_exhausted

    def next_batch(self):
        scene_samples = []
        current_raw_agents = 0
        aug_to_buffer = 0 # flag to push augmentation of large scene to buffer
        
        # Internal function: Try to add scene
        def try_add_scene(scene):
            nonlocal current_raw_agents
            n_agents = len(scene['pre_motion_3D'])
            
            if (current_raw_agents + n_agents) > self.effective_limit:
                return False
            scene_samples.append(scene)
            current_raw_agents += n_agents
            return True

        # --- Phase 1: Prioritize consuming Buffer ---
        while len(self.buffer) > 0 and len(scene_samples) < self.target_count:
            next_scene = self.buffer[0]
            if try_add_scene(next_scene):
                self.buffer.popleft() # Successfully added, remove from Buffer
            else: # case if scene is too large to fit in batch
                if len(scene_samples) == 0: # still fetch if batch is empty
                    scene_samples.append(self.buffer.popleft())
                    current_raw_agents += len(scene_samples[-1]['pre_motion_3D'])
                    # if not scene.get('is_augmented', False):
                    #     aug_to_buffer = 1
                # if batch is not empty, do not fetch this large scene
                break

        # --- Phase 2: Fetch new data from Generator ---
        while len(scene_samples) < self.target_count and current_raw_agents < self.effective_limit:
            # If agent limit is reached, stop fetching new ones
            # if current_raw_agents >= self.effective_limit:
            #     break

            # stop fetching if already have large scene, agent level is reached, or generator is exhausted
            if current_raw_agents >= self.agent_level or self.generator_exhausted: 
                break

            # epoch end
            if self.generator.is_epoch_end():
                self.generator_exhausted = True # Mark it, so has_data will return False afterwards
                break

            # Fetch data
            scene = self.generator()
            
            if scene is None: continue
            # if fail to fetch scene, push to buffer
            if not try_add_scene(scene):
                if len(scene_samples) == 0:
                    scene_samples.append(scene)
                    current_raw_agents += len(scene['pre_motion_3D'])
                    break
                self.buffer.append(scene)
                
                # break # End this round of fetching

        if len(scene_samples) == 0:
            return None

        # --- Augmentation ---
        if self.augment:
            new_augmented_samples = []
            new_aug_agents_count = 0

            for scene in scene_samples:
                # 1. Check if scene is already augmented (avoid infinite loop)
                if scene.get('is_augmented', False):
                    continue

                # 2. Generate augmented version
                aug_scene = copy.deepcopy(scene)
                k = np.random.randint(0, 24) 
                angle = k * (2 * np.pi / 24)
                aug_scene = rotate_scene(aug_scene, angle)
                aug_scene['is_augmented'] = True # Mark it!
                
                new_augmented_samples.append(aug_scene)
                new_aug_agents_count += len(aug_scene['pre_motion_3D'])

            # 3. Logic: Calculate total
            # If (original + new augmented) > Limit, store augmented version in Buffer
            if (current_raw_agents + new_aug_agents_count) > self.effective_limit:
                self.buffer.extend(new_augmented_samples)
            else:
                # If not exceeded, add to current Batch
                scene_samples.extend(new_augmented_samples)

        # --- Phase 4: Collate ---
        return collate_scenes(scene_samples, mask=len(scene_samples)>1, conn_dist=self.conn_dist)


def prepare_batch(batch, device):
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

    # Tranpose 
    pre_motion = pre_motion.transpose(0, 1).contiguous() # [Batch, Time, 2]X
    fut_motion = fut_motion.transpose(0, 1).contiguous()

    data['pre_motion'] = pre_motion
    data['fut_motion'] = fut_motion
    # Note: Scene centering is now handled in data_loader class to save compute resources
    # data['agent_current_pos'] = current_pos_t0 - scene_center # to recover original position, shape: [Agents, 2]
    data['agent_num'] = pre_motion.shape[1]
    data['seq_start_end'] = batch['seq_start_end']
    
    # ... (Rest of the function remains the same) ...
    
    # Create connectivity mask (Using 'pre_motion_mask' if available, else fully connected)
    if 'agent_mask' in batch:
        data['agent_mask'] = to_tensor(batch['agent_mask']).to(device)
    else:
        data['agent_mask'] = torch.zeros(data['agent_num'], data['agent_num']).to(device)

    data['heading'] = None # torch.zeros(data['agent_num']).to(device)
    data['heading_vec'] = None # torch.zeros(data['agent_num'], 2).to(device)
    pre_vel = torch.zeros_like(pre_motion)
    # pre_vel[1:] = pre_motion[1:] - pre_motion[:-1]
    data['pre_vel'] = None # pre_vel
    data['pre_motion_scene_norm'] = pre_motion
    data['agent_enc_shuffle'] = None
    
    if 'fut_motion_mask' in batch:
        mask = torch.stack([to_tensor(m) for m in batch['fut_motion_mask']], dim=0).to(device)
        data['fut_mask'] = mask.transpose(0, 1).contiguous()
    
    return data


def rotate_scene(scene, angle):
    """
    Do Random Rotation Augmentation
    """
    # rot mtx
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]]) # [2, 2]

    motion_keys = ['pre_motion_3D', 'fut_motion_3D'] 
    for key in motion_keys:
        if key in scene and scene[key] is not None:
            new_motion_list = []
            for agent_motion in scene[key]:
                agent_motion_arr = np.asarray(agent_motion) 
                rotated_motion = agent_motion_arr @ R.T
                
                new_motion_list.append(rotated_motion)
            scene[key] = new_motion_list
    return scene


def relative_to_abs(rel_traj, start_pos):
    # 1. Cumulative sum over time axis -> Calculate total displacement relative to start_pos
    rel_traj = torch.cumsum(rel_traj, dim=0)
    
    # 2. Add starting position (start_pos)
    # start_pos needs to be unsqueezed to [1, batch, 2] for broadcasting addition
    abs_traj = rel_traj + start_pos.unsqueeze(0)
    
    return abs_traj


def check_accuracy(args, loader, generator, limit=False, k=20, augment=False):
    """
    Evaluates the generator using Best-of-N strategy.
    Returns ADE, FDE, and L2 (MSE) of the best trajectory.
    
    Args:
        args: Arguments object
        loader: Data loader
        generator: Generator model
        limit: Whether to limit number of samples
        k: Number of samples for Best-of-K evaluation
        augment: Whether to apply data augmentation (rotation)
    """
    metrics = {}
    ade_outer, fde_outer = [], []
    
    # Initialize accumulators for global L2 calculation
    total_l2_error = 0.0
    total_valid_points = 0.0
    
    total_traj = 0
    
    generator.eval()
    
    # Use SmartBatcher instead of fetch_and_collate_batch
    conn_dist = getattr(args, 'conn_dist', 100000.0)
    batcher = SmartBatcher(loader, args.batch_size, augment=augment, max_agents_limit=50, conn_dist=conn_dist)
    batcher.reset()
    
    with torch.no_grad():
        while batcher.has_data():
            # 1. Prepare Batch and Generate K samples
            raw_batch = batcher.next_batch()
            if raw_batch is None: continue
            
            batch = prepare_batch(raw_batch, get_device(args))
            
            pred_fake_k, _ = generator(batch, k=k) # [K, Agents, Time, 2]
            pred_fake_k = pred_fake_k.permute(1, 0, 2, 3) # [K, Agents, Time, 2]
            
            # Process Ground Truth: [Time, Agents, 2]
            pred_real_abs = batch['fut_motion'].permute(1, 0, 2)
            # pred_real_k = pred_real_abs.unsqueeze(0).expand(k, -1, -1, -1)

            loss_mask = None
            if 'fut_mask' in batch:
                loss_mask = batch['fut_mask'].transpose(0, 1) # [Agents, Time]

            # 3. Use Helper to Pick Best Scene Trajectories
            best_pred_fake, batch_l2_sum = select_best_k_scene(
                pred_fake_k, 
                pred_real_abs, 
                batch['seq_start_end'], 
                loss_mask
            )
            
            # --- L2 Metric Accumulation ---=
            total_l2_error += batch_l2_sum.item()
            
            if loss_mask is not None:
                total_valid_points += loss_mask.sum().item()
            else:
                total_valid_points += (batch['agent_num'] * args.future_frames)

            # --- ADE / FDE Calculation (using the selected best_pred_fake) ---
            diff = best_pred_fake - pred_real_abs
            dist = torch.norm(diff, dim=-1) # [Agents, Time]
            
            if loss_mask is not None:
                # ADE: Average over Time (dim=1)
                ade = (dist * loss_mask).sum(dim=1) / (loss_mask.sum(dim=1) + 1e-6)
                # FDE: Last Time Step
                fde = dist[:, -1] * loss_mask[:, -1]
            else:
                ade = dist.mean(dim=1)
                fde = dist[:, -1]
                
            ade_outer.append(ade)
            fde_outer.append(fde)
            
            total_traj += batch['agent_num']
            
            if limit and total_traj >= args.num_samples_check:
                break

    # Final Aggregation
    ade_all = torch.cat(ade_outer).mean().item()
    fde_all = torch.cat(fde_outer).mean().item()
    
    # Calculate global Mean Squared Error
    l2_all = total_l2_error / (total_valid_points + 1e-6)
    
    metrics['ade'] = ade_all
    metrics['fde'] = fde_all
    metrics['l2'] = l2_all
    
    generator.train()
    return metrics


'''
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
'''

def load_config_from_yaml(config_path):
    """
    Load configuration from YAML file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config if config else {}
