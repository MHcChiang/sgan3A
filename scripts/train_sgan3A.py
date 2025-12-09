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
            # scene mask
            scene_mask = np.zeros((n_agents, n_agents), dtype=np.float32)
            big_mask[current_idx : current_idx+n_agents, 
                    current_idx : current_idx+n_agents] = scene_mask
            current_idx += n_agents
            
        batch_data['agent_mask'] = big_mask
    return batch_data


def fetch_and_collate_batch(generator, batch_size, augment=False, max_agents_limit=10):
    """
    Fetches scenes and applies Online Augmentation (Data Doubling). Agent-Cap Batching (Soft Limit Strategy)
    1. Fetches half batch if augment is True.
    2. Duplicates and rotates the fetched scenes.
    3. Collates everything into a single batch.
    """
    scene_samples = []
    total_agents_in_batch = 0
    
    # 1. define the number of scenes to fetch
    num_to_fetch = batch_size // 2 if augment else batch_size
    if num_to_fetch < 1: num_to_fetch = 1

    # 2. Fetch scenes with agent limit check
    while len(scene_samples) < num_to_fetch:
        if generator.is_epoch_end():
            break

        scene = generator()
        if scene is None: continue
        # add scene to scene_samples
        scene_samples.append(scene)
        # cal #agents
        current_scene_agents = len(scene['pre_motion_3D'])
        if augment:
            current_scene_agents *= 2
            
        total_agents_in_batch += current_scene_agents

        # check if total_agents_in_batch is greater than max_agents_limit
        if total_agents_in_batch > max_agents_limit:
            break

    if len(scene_samples) == 0:
        return None

    # 3. 執行 Online Augmentation (Data Doubling)
    if augment:
        augmented_samples = []
        for scene in scene_samples:
            # 深拷貝以避免修改到原始數據
            aug_scene = copy.deepcopy(scene)
            
            # Rotate by 2 * pi / 24 = pi / 12 (15 degrees interval)
            k = np.random.randint(0, 24) 
            angle = k * (2 * np.pi / 24)
        
            aug_scene = rotate_scene(aug_scene, angle)
            augmented_samples.append(aug_scene)
        
        scene_samples.extend(augmented_samples)

    # 4. Collate 
    # mask=True 代表會建立 Agent Mask (Block Diagonal)，這裡會自動處理變大後的 batch
    return collate_scenes(scene_samples, mask=len(scene_samples)>1)


class SmartBatcher:
    def __init__(self, generator, batch_size, augment=False, max_agents_limit=50):
        self.generator = generator
        self.augment = augment
        self.buffer = deque()
        
        # Q1優化: 初始化時就處理好 Augmentation 的係數
        if self.augment:
            self.target_count = max(1, batch_size // 2)
            self.effective_limit = max_agents_limit // 2
        else:
            self.target_count = batch_size
            self.effective_limit = max_agents_limit

        self.agent_level = int(0.75 * self.effective_limit)
        self.generator_exhausted = False
            
    def reset(self):
        """每個 Epoch 開始前呼叫"""
        self.buffer.clear()
        self.generator.shuffle()
        self.generator_exhausted = False 
        
    def has_data(self):
        """
        判斷是否還有數據。
        關鍵：這裡絕對不呼叫 generator.is_epoch_end()，只看 Buffer 和內部的 Flag。
        """
        if len(self.buffer) > 0:
            return True
        
        return not self.generator_exhausted

    def next_batch(self):
        scene_samples = []
        current_raw_agents = 0
        aug_to_buffer = 0 # flag to push augmentation of large scene to buffer
        
        # 內部函數：嘗試加入場景
        def try_add_scene(scene):
            nonlocal current_raw_agents
            n_agents = len(scene['pre_motion_3D'])
            
            if (current_raw_agents + n_agents) > self.effective_limit:
                return False
            scene_samples.append(scene)
            current_raw_agents += n_agents
            return True

        # --- 階段 1: 優先消化 Buffer ---
        while len(self.buffer) > 0 and len(scene_samples) < self.target_count:
            next_scene = self.buffer[0]
            # breakpoint()
            if try_add_scene(next_scene):
                self.buffer.popleft() # 成功加入，移出 Buffer
            else: # case if scene is too large to fit in batch
                if len(scene_samples) == 0: # still fetch if batch is empty
                    scene_samples.append(self.buffer.popleft())
                    current_raw_agents += len(scene_samples[-1]['pre_motion_3D'])
                    # if not scene.get('is_augmented', False):
                    #     aug_to_buffer = 1
                # if batch is not empty, do not fetch this large scene
                break

        # --- 階段 2: 從 Generator 獲取新數據 ---
        while len(scene_samples) < self.target_count and current_raw_agents < self.effective_limit:
            # 如果已經達到 Agent 限制，就不再抓新的
            # if current_raw_agents >= self.effective_limit:
            #     break

            # stop fetching if already have large scene, agent level is reached, or generator is exhausted
            if current_raw_agents >= self.agent_level or self.generator_exhausted: 
                break

            # epoch end
            if self.generator.is_epoch_end():
                self.generator_exhausted = True # 標記起來，之後 has_data 就會回傳 False
                break

            # 獲取數據
            scene = self.generator()
            
            if scene is None: continue
            # if fail to fetch scene, push to buffer
            if not try_add_scene(scene):
                if len(scene_samples) == 0:
                    scene_samples.append(scene)
                    current_raw_agents += len(scene['pre_motion_3D'])
                    break
                self.buffer.append(scene)
                
                # break # 結束這一輪 Fetch
        # breakpoint()
        if len(scene_samples) == 0:
            return None

        # --- Augmentation ---
        if self.augment:
            new_augmented_samples = []
            new_aug_agents_count = 0

            for scene in scene_samples:
                # 1. 檢查是否已經是增強過的場景 (避免無限循環)
                if scene.get('is_augmented', False):
                    continue

                # 2. 產生增強版
                aug_scene = copy.deepcopy(scene)
                k = np.random.randint(0, 24) 
                angle = k * (2 * np.pi / 24)
                aug_scene = rotate_scene(aug_scene, angle)
                aug_scene['is_augmented'] = True # 標記它！
                
                new_augmented_samples.append(aug_scene)
                new_aug_agents_count += len(aug_scene['pre_motion_3D'])

            # 3. 你的邏輯：算總帳
            # 如果 (原本的 + 新增強的) > Limit，就把增強版存 Buffer
            if (current_raw_agents + new_aug_agents_count) > self.effective_limit:
                self.buffer.extend(new_augmented_samples)
            else:
                # 沒爆，就加入當前 Batch
                scene_samples.extend(new_augmented_samples)

        # debug
        # if current_raw_agents > self.effective_limit:
        #     print(f"num of scene_samples: {len(scene_samples)}")
        #     for i, scene in enumerate(scene_samples):
        #         print(f"scene{i}, # agents: {len(scene['pre_motion_3D'])}")
        #     # for scene in augmented_samples:
        #     #     print(f"augmented scene agents: {len(scene['pre_motion_3D'])}")
        #     breakpoint()
        # --- 階段 4: Collate ---
        return collate_scenes(scene_samples, mask=len(scene_samples)>1)


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
    
    # 1206 ADD: scene-centered coordinate system (Agentformer sec4-Implementation detail)
    current_pos_t0 = pre_motion[-1]
    scene_center = torch.mean(current_pos_t0, dim=0, keepdim=True) 
    pre_motion = pre_motion - scene_center
    fut_motion = fut_motion - scene_center

    data['pre_motion'] = pre_motion
    data['fut_motion'] = fut_motion
    data['scene_center'] = scene_center
    # data['agent_current_pos'] = current_pos_t0 - scene_center # to recover original position, shape: [Agents, 2]
    data['agent_num'] = pre_motion.shape[1]
    
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
    # 1. 對時間軸做累加 (Cumulative Sum) -> 算出相對於 start_pos 的總位移
    rel_traj = torch.cumsum(rel_traj, dim=0)
    
    # 2. 加上起始位置 (start_pos)
    # start_pos 需要 unsqueeze 變成 [1, batch, 2] 才能做廣播加法
    abs_traj = rel_traj + start_pos.unsqueeze(0)
    
    return abs_traj


# # Helper to rotate a tensor [..., 2]
# def rotate_tensor(t, c, s):
#     # t shape: [..., 2]
#     x = t[..., 0]
#     y = t[..., 1]
#     x_new = x * c - y * s
#     y_new = x * s + y * c
#     return torch.stack([x_new, y_new], dim=-1)


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

    batcher = SmartBatcher(train_gen, args.batch_size, augment=args.augment, max_agents_limit=50)
    logger.info(f"Used SmartBatcher to fetch batch, batch size: {args.batch_size}, agent limit: {batcher.effective_limit*2}")
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
                    losses_d = discriminator_step(args, batch, generator, discriminator, gan_d_loss, optimizer_d, scaler, device)
            else:
                losses_d = {'D_loss': 0.0}
            
            for _ in range(args.g_steps):
                losses_g = generator_step(args, batch, generator, discriminator, gan_g_loss, optimizer_g, scaler, device, is_warmup=is_warmup)
            
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
        val_metrics = check_accuracy(args, val_gen, generator, limit=True, k=1, augment=False)
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


def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d, scaler, device):
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
    
    # 4. Gradient Clipping (必須先 Unscale)
    if args.clipping_threshold_d > 0:
        scaler.unscale_(optimizer_d)
        nn.utils.clip_grad_norm_(discriminator.parameters(), args.clipping_threshold_d)
    
    # 5. Optimizer Step with Scaler
    scaler.step(optimizer_d)
    scaler.update()

    return losses


# def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g, scaler, device, is_warmup=False):
#     """
#     Generator optimization step with optional Best-of-K (Variety Loss).
    
#     Args:
#         k (int): Number of samples to generate for Variety Loss. 
#                  If k=1, standard GAN training.
#                  If k>1, minimizes L2 error of the best sample among k generations.
#     """
#     losses = {}
#     k = args.k
    
#     optimizer_g.zero_grad()

#     # 開啟混合精度環境
#     # with torch.amp.autocast('cuda'):
#     # with torch.cuda.amp.autocast(enabled=(args.device.type == 'cuda')):
#     with torch.cuda.amp.autocast(enabled=False):
#         loss = torch.zeros(1, device=device)
        
#         # Ground Truth
#         pred_real_abs = batch['fut_motion'] # [Time, Agents, 2]
#         loss_mask = batch.get('fut_mask', None)
#         if loss_mask is not None:
#             loss_mask = loss_mask.transpose(0, 1)

#         # --- Forward Generator ---
#         if k == 1:
#             pred_fake_abs, data_dict = generator(batch)
#             pred_fake_abs = pred_fake_abs.permute(1, 0, 2)
#             best_pred_fake = pred_fake_abs
#             best_data_dict = data_dict
#         else:
#             # Variety Loss Logic (Best-of-K)
#             preds_k = []
#             data_dicts_k = []
#             for _ in range(k):
#                 p, d = generator(batch)
#                 p = p.permute(1, 0, 2)
#                 preds_k.append(p)
#                 data_dicts_k.append(d)
                
#             stack_preds = torch.stack(preds_k, dim=0) 
            
#             # L2 Error Calculation
#             diff = stack_preds - pred_real_abs.unsqueeze(0)
#             dist_sq = diff.pow(2).sum(dim=-1)
            
#             if loss_mask is not None:
#                 mask_reshaped = loss_mask.transpose(0, 1).unsqueeze(0)
#                 loss_dist = (dist_sq * mask_reshaped).sum(dim=1)
#             else:
#                 loss_dist = dist_sq.sum(dim=1)
                
#             min_vals, min_inds = loss_dist.min(dim=0)
            
#             # Gather Best Trajectories
#             agents_num = stack_preds.shape[2]
#             best_pred_list = []
#             for i in range(agents_num):
#                 best_idx = min_inds[i].item()
#                 best_pred_list.append(stack_preds[best_idx, :, i, :])
#             best_pred_fake = torch.stack(best_pred_list, dim=1)
#             best_data_dict = data_dicts_k[0]

#         # --- Losses Calculation ---
#         l2 = l2_loss(best_pred_fake, pred_real_abs, loss_mask, mode='average')
#         loss = loss + args.l2_loss_weight * l2  # Use standard addition to avoid inplace errors
#         losses['G_l2'] = l2.item()

#         if args.use_cvae:
#             q_dist = best_data_dict['q_z_dist']
#             p_dist = best_data_dict['p_z_dist_infer']
#             kl = torch.distributions.kl.kl_divergence(q_dist, p_dist).sum(dim=-1).mean()
#             loss = loss + args.kl_weight * kl
#             losses['G_kl'] = kl.item()

#         if not is_warmup:
#             scores_fake = discriminator(batch['pre_motion'], best_pred_fake, batch['agent_mask'], batch['agent_num'])
#             loss_adv = g_loss_fn(scores_fake)
#             loss = loss + loss_adv
#             losses['G_adv'] = loss_adv.item()
#         else:
#             losses['G_adv'] = 0.0

#     # Backward & Step with Scaler
#     scaler.scale(loss).backward()
    
#     if args.clipping_threshold_g > 0:
#         scaler.unscale_(optimizer_g)
#         nn.utils.clip_grad_norm_(generator.parameters(), args.clipping_threshold_g)
        
#     scaler.step(optimizer_g)
#     scaler.update()

#     return losses


def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g, scaler, device, is_warmup=False):
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
        # 1. 將 GT 轉為 [Batch, Time, 2] 以便統一計算
        pred_real_norm = batch['fut_motion'].permute(1, 0, 2) 
        
        # 2. 準備當前位置用於還原 (Residual Connection)
        # [Batch, 2] -> [Batch, 1, 2] (方便廣播加到 Time 維度)
        # agent_current_pos = batch['agent_current_pos'].unsqueeze(1) 
        
        loss_mask = batch.get('fut_mask', None)
        if loss_mask is not None:
            # [Time, agents] -> 
            loss_mask = loss_mask.transpose(0, 1)   # [agents, Time]
        else:
            loss_dist = dist_sq.sum(dim=2)

        # --- Forward Generator ---
        if k == 1:
            pred_fake_offset, data_dict = generator(batch) # Output: [agents, time, 2]
            
            # 還原座標: [Batch, 1, 2] + [Batch, Time, 2]
            pred_fake_norm = pred_fake_offset  #+  agent_current_pos
            
            best_pred_fake = pred_fake_norm
            best_data_dict = data_dict
        else:
            # Variety Loss Logic (Best-of-K)
            preds_k = []
            data_dicts_k = []
            for _ in range(k):
                p_norm, d = generator(batch) # Output: [agents, time, 2]
                # p_norm = agent_current_pos + p_offset
                preds_k.append(p_norm)
                data_dicts_k.append(d)
                
            stack_preds = torch.stack(preds_k, dim=0) # [K, Agents, Time, 2]
            
            # L2 Error Calculation (Find best k)
            diff = stack_preds - pred_real_norm.unsqueeze(0) # [K, Agents, Time, 2]
            dist_sq = diff.pow(2).sum(dim=-1) # [K, Agents, Time]

            masked_dist = dist_sq * loss_mask # [K, Agents, Time]

            loss_dist = masked_dist.sum(dim=2) # [K, Agents]
            
            min_vals, min_inds = loss_dist.min(dim=0) # Min over K -> [Agents]

            # Gather Best Trajectories
            batch_size = stack_preds.shape[1]
            best_pred_list = []
            for i in range(batch_size):
                best_idx = min_inds[i].item()
                # Select: [K, B, T, 2] -> [T, 2]
                best_pred_list.append(stack_preds[best_idx, i, :, :])
            
            # Stack back to [Batch, Time, 2]
            best_pred_fake = torch.stack(best_pred_list, dim=0)
            best_data_dict = data_dicts_k[0]

        # --- Losses Calculation ---
        best_pred_fake = best_pred_fake.permute(1, 0, 2)
        pred_real_norm = pred_real_norm.permute(1, 0, 2)
        # breakpoint()
        l2 = l2_loss(best_pred_fake, pred_real_norm, loss_mask, mode='average')
        
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
            # scores_fake = discriminator(batch['pre_motion'], best_pred_fake, batch['agent_mask'], batch['agent_num'])
            scores_fake = discriminator(batch['pre_motion'], best_pred_fake, batch['agent_mask'], batch['agent_num'])
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
    l2_outer = [] 
    total_traj = 0
    
    generator.eval()
    
    with torch.no_grad():
        while not loader.is_epoch_end():
            # 1. Fetch & Collate (with optional augmentation)
            raw_batch = fetch_and_collate_batch(loader, args.batch_size, augment=augment)
            if raw_batch is None: continue
            
            # 2. Convert to Tensors
            batch = prepare_batch(raw_batch, get_device(args))
            
            # 3. Multiple Sampling (Best-of-N)
            pred_fake_list = []
            for _ in range(k):
                # 假設 generator 輸出是 [Agents, Time, 2]
                pred_fake_abs, _ = generator(batch) 
                pred_fake_list.append(pred_fake_abs)
            
            # [K, Agents, Time, 2]
            pred_fake_k = torch.stack(pred_fake_list, dim=0) 
            
            # 處理 Ground Truth: [Time, Agents, 2] -> [Agents, Time, 2] -> [K, Agents, Time, 2]
            pred_real_abs = batch['fut_motion'].permute(1, 0, 2)
            pred_real_k = pred_real_abs.unsqueeze(0).expand(k, -1, -1, -1)
            
            # 4. Calculate Difference
            diff = pred_fake_k - pred_real_k
            
            # dist shape: [K, Agents, Time] (因為 dim=-1 把座標 (x,y) 算掉了)
            dist = torch.norm(diff, dim=-1) 
            dist_sq = diff.pow(2).sum(dim=-1)
            
            # Handling Valid Mask
            if 'fut_mask' in batch:
                # batch['fut_mask'] 原始通常是 [Time, Agents]
                # 我們需要轉置成 [Agents, Time] 來配合 dist
                valid_mask = batch['fut_mask'].transpose(0, 1) > 0 
                valid_mask_k = valid_mask.unsqueeze(0) # [1, Agents, Time]
                
                # --- ADE Calculation ---
                # 現在 Time 是 dim=2，所以我們要對 dim=2 求和
                # [K, Agents, Time] -> sum(dim=2) -> [K, Agents]
                ade_k = (dist * valid_mask_k).sum(dim=2) / (valid_mask_k.sum(dim=2) + 1e-6)
                
                # --- FDE Calculation ---
                # 取最後一個時間點 (Time is dim 2, so index -1 at axis 2)
                fde_k = dist[:, :, -1] * valid_mask_k[:, :, -1]
                
                # --- L2 (MSE) Calculation ---
                l2_k = (dist_sq * valid_mask_k).sum(dim=2) / (valid_mask_k.sum(dim=2) + 1e-6)
                
            else:
                # 如果沒有 Mask，直接對 Time (dim=2) 取平均
                ade_k = dist.mean(dim=2) # [K, Agents]
                fde_k = dist[:, :, -1]   # [K, Agents]
                l2_k = dist_sq.mean(dim=2) # [K, Agents]
            
            # 5. Best-of-N Selection (保持不變，因為 input 已經是 [K, Agents])
            best_ade, best_idx = ade_k.min(dim=0) # [Agents]
            best_fde, _ = fde_k.min(dim=0)        # [Agents]
            best_l2, _ = l2_k.min(dim=0)          # [Agents]
            
            ade_outer.append(best_ade)
            fde_outer.append(best_fde)
            l2_outer.append(best_l2)
            
            total_traj += batch['agent_num']
            
            if limit and total_traj >= args.num_samples_check:
                break

    ade_all = torch.cat(ade_outer).mean().item()
    fde_all = torch.cat(fde_outer).mean().item()
    l2_all = torch.cat(l2_outer).mean().item()
    
    metrics['ade'] = ade_all
    metrics['fde'] = fde_all
    metrics['l2'] = l2_all
    
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
