"""
Data loader that uses AgentFormer's data format directly.
Adapts SGAN training to work with AgentFormer's data structure.
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional

from .dataloader import data_generator


class AgentFormerDataset(Dataset):
    """
    Dataset that provides data in AgentFormer's format.
    Each sample is a dictionary with AgentFormer's data structure.
    """
    
    def __init__(self, args, data_paths, split='train'):
        """
        Args:
            args: Training arguments (needs obs_len, pred_len, etc.)
            data_paths: Path(s) to data files (can be single path or list)
            split: 'train', 'val', or 'test'
        """
        self.args = args
        self.split = split
        
        # Create a parser-like object for AgentFormer's data_generator
        class Parser:
            def __init__(self, args, data_paths):
                # Map SGAN args to AgentFormer parser format
                self.past_frames = args.obs_len
                self.min_past_frames = args.obs_len
                self.future_frames = args.pred_len
                self.min_future_frames = args.pred_len
                self.frame_skip = getattr(args, 'skip', 1)
                
                # Determine dataset name from path
                if isinstance(data_paths, list):
                    path = data_paths[0] if data_paths else ''
                else:
                    path = data_paths
                
                # Extract dataset name from path
                path_parts = path.split('/')
                if 'eth_ucy' in path_parts:
                    idx = path_parts.index('eth_ucy')
                    if idx + 1 < len(path_parts):
                        dataset_name = path_parts[idx + 1]
                        if dataset_name in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                            self.dataset = dataset_name
                        else:
                            self.dataset = 'eth'  # Default
                    else:
                        self.dataset = 'eth'
                else:
                    # Fallback: try to infer from path
                    if 'eth' in path.lower() and 'hotel' not in path.lower():
                        self.dataset = 'eth'
                    elif 'hotel' in path.lower():
                        self.dataset = 'hotel'
                    elif 'univ' in path.lower():
                        self.dataset = 'univ'
                    elif 'zara1' in path.lower():
                        self.dataset = 'zara1'
                    elif 'zara2' in path.lower():
                        self.dataset = 'zara2'
                    else:
                        self.dataset = 'eth'  # Default
                
                # Set data root paths
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                self.data_root_ethucy = os.path.join(project_root, 'datasets', 'eth_ucy')
                self.data_root_nuscenes_pred = os.path.join(project_root, 'datasets', 'nuscenes_pred')
            
            def get(self, key, default):
                return getattr(self, key, default)
        
        parser = Parser(args, data_paths)
        
        # Create AgentFormer data generator
        import logging
        log = logging.getLogger(__name__)
        phase = 'training' if split == 'train' else 'testing'
        self.data_gen = data_generator(parser, log, split=split, phase=phase)
        
        # Cache all samples
        self.samples = []
        self._load_all_samples()
    
    def _load_all_samples(self):
        """Load all samples from AgentFormer's data generator."""
        self.data_gen.shuffle()
        self.data_gen.index = 0
        
        while not self.data_gen.is_epoch_end():
            sample = self.data_gen.next_sample()
            if sample is not None:
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples for {self.split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Return data in AgentFormer's format.
        
        Returns:
            Dictionary with AgentFormer's data structure:
            - pre_motion_3D: list of (T, 2) numpy arrays
            - fut_motion_3D: list of (T, 2) numpy arrays
            - pre_motion_mask: list of (T,) bool arrays
            - fut_motion_mask: list of (T,) bool arrays
            - heading: list of floats or None
            - traj_scale: float
            - scene_map: map object or None
        """
        return self.samples[idx]


def data_loader(args, data_paths):
    """
    Create data loader that provides AgentFormer's data format.
    
    Args:
        args: Training arguments
        data_paths: Path(s) to data files (can be single path or list)
    
    Returns:
        dataset, dataloader tuple
    """
    # Determine split from context
    split = 'train' if 'train' in str(data_paths).lower() else 'val'
    
    dataset = AgentFormerDataset(args, data_paths, split=split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == 'train'),
        num_workers=args.loader_num_workers,
        collate_fn=collate_fn_agentformer
    )
    
    return dataset, dataloader


def collate_fn_agentformer(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for AgentFormer's data format.
    Combines multiple scenes into a batch, keeping AgentFormer's structure.
    
    Args:
        batch: List of AgentFormer data dictionaries
    
    Returns:
        Batched data dictionary in AgentFormer's format
    """
    # Collect all agents from all scenes
    pre_motion_3D_batch = []
    fut_motion_3D_batch = []
    pre_motion_mask_batch = []
    fut_motion_mask_batch = []
    heading_batch = []
    
    # Use first sample's metadata (assuming same across batch)
    traj_scale = batch[0]['traj_scale']
    scene_map = batch[0].get('scene_map', None)
    
    for sample in batch:
        pre_motion_3D_batch.extend(sample['pre_motion_3D'])
        fut_motion_3D_batch.extend(sample['fut_motion_3D'])
        pre_motion_mask_batch.extend(sample['pre_motion_mask'])
        fut_motion_mask_batch.extend(sample['fut_motion_mask'])
        if sample['heading'] is not None:
            heading_batch.extend(sample['heading'])
        else:
            heading_batch.extend([None] * len(sample['pre_motion_3D']))
    
    # Create seq_start_end to track which agents belong to which scene
    seq_start_end = []
    current_idx = 0
    for sample in batch:
        num_agents = len(sample['pre_motion_3D'])
        if num_agents > 0:
            seq_start_end.append((current_idx, current_idx + num_agents))
            current_idx += num_agents
    
    batched_data = {
        'pre_motion_3D': pre_motion_3D_batch,
        'fut_motion_3D': fut_motion_3D_batch,
        'pre_motion_mask': pre_motion_mask_batch,
        'fut_motion_mask': fut_motion_mask_batch,
        'heading': heading_batch if any(h is not None for h in heading_batch) else None,
        'traj_scale': traj_scale,
        'scene_map': scene_map,
        'seq_start_end': seq_start_end,  # Added for convenience
    }
    
    return batched_data
