import os
import sys
import argparse
import time
import numpy as np
import torch
from torch import optim

sys.path.append(os.getcwd())

# Import your modules
# Ensure these are available in your python path
from model.sgan3A import AgentFormerGenerator, AgentFormerDiscriminator
from model.data.dataloader import data_generator

# ==========================================
# 0. Dummy Logger (Fixes 'NoneType' error)
# ==========================================
class Logger:
    def info(self, msg):
        print(msg)
    def write(self, msg):
        print(msg)
    def flush(self):
        pass
    def close(self):
        pass
    def __del__(self):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    def __enter__(self):
        return self

# ==========================================
# 1. Configuration & Logger
# ==========================================
class Config:
    def __init__(self):
        # --- Experiment Params ---
        self.batch_size = 32  # Number of scenes to stack per batch
        
        # --- Dataset Params ---
        self.dataset = 'eth' 
        self.data_root_ethucy = 'datasets/eth_ucy'
        self.data_root_nuscenes_pred = 'datasets/nuscenes_pred'
        self.traj_scale = 2
        
        self.past_frames = 8
        self.future_frames = 12
        self.min_past_frames = 8
        self.min_future_frames = 12
        self.frame_skip = 1
        self.phase = 'training'
        self.split = 'train'

        # --- Model Params ---
        self.motion_dim = 2
        self.forecast_dim = 2
        self.tf_model_dim = 256
        self.tf_nhead = 8
        self.tf_ff_dim = 512
        self.tf_dropout = 0.1
        self.context_encoder = {'nlayer': 2}
        self.future_decoder = {'nlayer': 2, 'out_mlp_dim': [512, 256]}
        self.pos_concat = True
        self.nz = 32
        self.z_type = 'gaussian'
        self.nlayer = 2 
        
    def get(self, key, default=None): return getattr(self, key, default)


# ==========================================
# 2. Batch Construction Helpers
# ==========================================

def collate_scenes(scenes):
    """
    Merges a list of scene dictionaries into a single batch dictionary.
    Handles missing keys and calculates agent counts dynamically.
    """
    batch_data = {}
    
    # 1. Merge Lists of Trajectories: past / future / mask / heading
    # Mask: indicate which timesteps contain valid data for each agent (Some agents might enter the scene late or leave early.)
    # Heading: The direction the agent is facing at the last observed timestep
    keys_to_merge = ['pre_motion_3D', 'fut_motion_3D', 'pre_motion_mask', 'fut_motion_mask', 'heading']
    
    for key in keys_to_merge:
        # Check if key exists and is not None in the first sample
        if scenes[0].get(key) is not None:
            merged_list = []
            for s in scenes:
                merged_list.extend(s[key])
            batch_data[key] = merged_list
        else:
            batch_data[key] = None

    # 2. Sum Agent Counts (FIXED: Calculate len instead of looking up key)
    # The number of agents is simply the length of the motion list
    agents_per_scene = [len(s['pre_motion_3D']) for s in scenes]
    total_agents = sum(agents_per_scene)
    batch_data['agent_num'] = total_agents

    # 3. Create Block-Diagonal Mask
    # Initialize giant mask with -inf (Block everything)
    big_mask = np.full((total_agents, total_agents), float('-inf'), dtype=np.float32)
    
    current_idx = 0
    for n_agents in agents_per_scene:
        # Create Local Mask for THIS scene (Fully Connected = 0.0)
        # Note: If you have a specific connectivity mask from the loader, merge it here.
        # Otherwise, 0.0 means all agents in this scene see each other.
        scene_mask = np.zeros((n_agents, n_agents), dtype=np.float32)
        
        # Place Local Mask into the Block Diagonal
        big_mask[current_idx : current_idx+n_agents, 
                 current_idx : current_idx+n_agents] = scene_mask
                 
        current_idx += n_agents
        
    batch_data['agent_mask'] = big_mask
    return batch_data

def fetch_and_collate_batch(generator, batch_size):
    """
    Fetches 'batch_size' scenes from the generator and collates them into one batch.
    Returns: Collated batch dictionary (numpy) or None if generator is empty.
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
        print(f"Collating {len(scene_samples)} scenes into batch...")
        return collate_scenes(scene_samples)
    else:
        return None

def prepare_batch(batch, device):
    """Converts numpy batch to tensor batch on device."""
    def to_tensor(x):
        if isinstance(x, torch.Tensor): return x.clone().detach().float()
        elif isinstance(x, np.ndarray): return torch.from_numpy(x).float()
        else: return torch.tensor(x).float()

    data = {}
    
    # Stack lists into Tensors [Time, Agents, 2]
    pre_motion = torch.stack([to_tensor(m) for m in batch['pre_motion_3D']], dim=0).to(device)
    pre_motion = pre_motion.transpose(0, 1).contiguous() 
    
    fut_motion = torch.stack([to_tensor(m) for m in batch['fut_motion_3D']], dim=0).to(device)
    fut_motion = fut_motion.transpose(0, 1).contiguous()

    data['pre_motion'] = pre_motion
    data['fut_motion'] = fut_motion
    data['agent_num'] = pre_motion.shape[1]
    data['agent_mask'] = to_tensor(batch['agent_mask']).to(device)
    
    # Helpers
    data['heading'] = torch.zeros(data['agent_num']).to(device)
    data['heading_vec'] = torch.zeros(data['agent_num'], 2).to(device)
    data['pre_vel'] = torch.zeros_like(pre_motion)
    data['pre_motion_scene_norm'] = pre_motion
    data['agent_enc_shuffle'] = None

    return data

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == '__main__':
    cfg = Config()
    log = Logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = AgentFormerGenerator(cfg).to(device)
    discriminator = AgentFormerDiscriminator(cfg).to(device)
    
    try:
        loader = data_generator(cfg, log, split='train', phase='training')
        print(f"Loader initialized. Dataset size: {loader.num_total_samples}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Fetching batch of {cfg.batch_size}...")
    
    # 1. Fetch & Collate (Numpy)
    raw_batch = fetch_and_collate_batch(loader, cfg.batch_size)
    print(f"Big Mask Shape: {raw_batch['agent_mask'].shape}")
    print(f"Big Mask:\n{raw_batch['agent_mask']}")

    # print("Fetching next batch of scenes...")
    # raw_batch = fetch_and_collate_batch(loader, cfg.batch_size)
    # print(f"Big Mask Shape: {raw_batch['agent_mask'].shape}")
    # print(f"Big Mask:\n{raw_batch['agent_mask']}")
    
    if raw_batch is not None:
        # 2. Prepare (Tensor)
        data = prepare_batch(raw_batch, device)
        
        print(f"Batch Loaded.")
        print(f"Total Agents in Batch: {data['agent_num']}")
        print(f"Block Mask Shape: {data['agent_mask'].shape}")

        # 3. Forward Pass
        print("Running Generator...")
        with torch.no_grad():
            pred_fake = generator(data)
        print(f"Generator Output: {pred_fake.shape}")

        print("Running Discriminator...")
        score = discriminator(data['pre_motion'], pred_fake, data['agent_mask'], data['agent_num'])
        print(f"Discriminator Score: {score.shape}")
    else:
        print("Loader empty.")