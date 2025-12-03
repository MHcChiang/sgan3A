import math
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from .agentformer_lib import (AgentFormerEncoderLayer, AgentFormerDecoderLayer,AgentFormerEncoder, AgentFormerDecoder,)
from .agentformer import PositionalAgentEncoding, ContextEncoder, FutureDecoder


def get_noise(shape, noise_type='gaussian', device='cpu'):
    if noise_type == 'gaussian':
        return torch.randn(shape, device=device)
    if noise_type == 'uniform':
        return torch.rand(shape, device=device) * 2.0 - 1.0
    raise ValueError(f'Unsupported noise type: {noise_type}')


def generate_mask(tgt_sz: int, src_sz: int, agent_num: int, agent_mask: torch.Tensor):
    assert tgt_sz % agent_num == 0 and src_sz % agent_num == 0
    return agent_mask.repeat(tgt_sz // agent_num, src_sz // agent_num)


def generate_ar_mask(sz: int, agent_num: int, agent_mask: torch.Tensor):
    assert sz % agent_num == 0
    mask = agent_mask.repeat(sz // agent_num, sz // agent_num)
    steps = sz // agent_num
    for t in range(steps - 1):
        i1 = t * agent_num
        i2 = (t + 1) * agent_num
        mask[i1:i2, i2:] = float('-inf')
    return mask


class AgentFormerGenerator(nn.Module):
    def __init__(self, cfg):
        """
        Adapts AgentFormer modules into a GAN Generator.
        Args:
            cfg: The global configuration object (same as used in train.py)
        """
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cpu')

        # -------------------------------------------------------
        # 1. Setup Context (ctx)
        # Replicate the 'ctx' setup from the AgentFormer class so the sub-modules know dimensions, input types, etc.
        # -------------------------------------------------------
        self.use_cvae = getattr(cfg, 'use_cvae', False)

        input_type = getattr(cfg, 'input_type', 'pos')
        pred_type = getattr(cfg, 'pred_type', input_type)
        print(f"input_type: {input_type}, pred_type: {pred_type}")
        if type(input_type) == str: input_type = [input_type]
        fut_input_type = getattr(cfg, 'fut_input_type', input_type)
        dec_input_type = getattr(cfg, 'dec_input_type', [])

        self.ctx = {
            'tf_cfg': getattr(cfg, 'tf_cfg', {}),
            'nz': cfg.nz, # Dimension of Latent Noise Z
            'z_type': cfg.z_type,
            'future_frames': cfg.future_frames,
            'past_frames': cfg.past_frames,
            'motion_dim': cfg.motion_dim,
            'forecast_dim': cfg.forecast_dim,
            'input_type': input_type,
            'fut_input_type': fut_input_type,
            'dec_input_type': dec_input_type,
            'pred_type': pred_type,
            'tf_nhead': cfg.tf_nhead,
            'tf_model_dim': cfg.tf_model_dim,
            'tf_ff_dim': cfg.tf_ff_dim,
            'tf_dropout': cfg.tf_dropout,
            'pos_concat': cfg.pos_concat,
            'ar_detach': False, # Usually False for GAN backprop!
            'max_agent_len': getattr(cfg, 'max_agent_len', 128),
            'use_agent_enc': getattr(cfg, 'use_agent_enc', False),
            'agent_enc_learn': getattr(cfg, 'agent_enc_learn', False),
            'agent_enc_shuffle': getattr(cfg, 'agent_enc_shuffle', False),
            'sn_out_type': getattr(cfg, 'sn_out_type', 'scene_norm'),
            'sn_out_heading': getattr(cfg, 'sn_out_heading', False),
            'vel_heading': getattr(cfg, 'vel_heading', False),
            'learn_prior': False, 
            'use_map': getattr(cfg, 'use_map', False),
            'context_dim': getattr(cfg, 'tf_model_dim', 256),
            'forecast_dim': getattr(cfg, 'forecast_dim', 2)
        }
        
        # -------------------------------------------------------
        # 2. Initialize Modules
        # -------------------------------------------------------
        # A. Context Encoder (Processes History X -> Context C)
        self.context_encoder = ContextEncoder(cfg.context_encoder, self.ctx)
        # breakpoint()
        # B. Future Decoder (Processes Context C + Noise Z -> Future Y)
        self.future_decoder = FutureDecoder(cfg.future_decoder, self.ctx)

    def forward(self, data, z=None):
        """
        Args:
            data: Dictionary containing 'pre_motion', 'agent_mask', etc.
            z: Latent noise vector [Batch_Size, agent_num, nz]. 
               If None, it is sampled from N(0,1).
        Returns:
            pred_fake: The generated future trajectories.
        """
        
        # 1. Encode Past: This populates data['context_enc'] and data['agent_context']
        self.context_encoder(data)

        # 2. Handle Latent Noise (Z)
        if self.use_cvae and self.training:
            # CVAE: Encode Future to get Posterior Q(z|X,Y)
            # This populates data['q_z_dist'] and data['q_z_samp']
            self.future_encoder(data)
            z = data['q_z_samp']
        else:
            if z is None:
                # Sample standard gaussian noise for GAN
                z = torch.randn(data['agent_num'], self.cfg.nz).to(data['pre_motion'].device)

        # 3. Decode Future (Generation) by x^{T} and latent vector
        # We use mode='infer' to generate future trajectories from the latent noise z
        # 'sample_num' is usually 1 for GAN training (1-to-1 mapping of Z to Y)
        # self.future_decoder(data, mode='infer', sample_num=1, autoregress=True, z=z)
        self.future_decoder(data, mode='infer', autoregress=True, z=z)

        # 4. Return Output
        # data['infer_dec_motion'] contains the predicted trajectory
        # Shape: [Batch_Size, Sample_Num, Frames, Dim]
        # pred_fake = data['infer_dec_motion']
        
        pred = data['infer_dec_motion'].squeeze(1)
        return pred, data


class AgentFormerDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        
        # 1. Dimensions
        self.model_dim = cfg.tf_model_dim
        self.ff_dim = cfg.tf_ff_dim
        self.nhead = cfg.tf_nhead
        self.dropout = cfg.tf_dropout
        self.nlayer = getattr(cfg, 'nlayer', 2) # Discriminator can often be shallower than Generator
        self.input_dim = cfg.motion_dim # Usually 2 (x, y)
        
        # 2. Input Projection
        # Projects (x,y) coordinates to Model Dimension
        self.input_fc = nn.Linear(self.input_dim, self.model_dim)

        # 3. Time Encoding
        # We reuse the class from agentformer.py
        self.pos_encoder = PositionalAgentEncoding(
            self.model_dim, 
            self.dropout, 
            concat=cfg.pos_concat
        )

        # 4. The Transformer Backbone (AgentFormer Encoder)
        # We construct this manually using the Lib
        encoder_layers = AgentFormerEncoderLayer(
            getattr(cfg, 'tf_cfg', {}), 
            self.model_dim, 
            self.nhead, 
            self.ff_dim, 
            self.dropout
        )
        self.tf_encoder = AgentFormerEncoder(encoder_layers, self.nlayer)

        # 5. Final Classifier (MLP)
        # We pool the features and map to a single score
        self.out_mlp = nn.Sequential(
            nn.Linear(self.model_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, history, future, agent_mask, agent_num):
        """
        Args:
            history: Past Trajectories [Batch, Past_Len, 2]
            future:  Future Trajectories [Batch, Fut_Len, 2] (Real or Fake)
            agent_mask: Connectivity mask (from data dict)
            agent_num: Number of agents
        """
        
        # 1. Concatenate History and Future along Time Dimension
        # future = future.permute(1, 0, 2)
        # Concatenate: [Total_Len, Total_Agents, 2]
        full_traj = torch.cat([history, future], dim=0)
        seq_len, batch_size, _ = full_traj.shape
        
        # 2. Input Projection
        # Flatten to [Batch*Total_Len, Input_Dim] for Linear Layer
        tf_in = self.input_fc(full_traj.view(-1, self.input_dim))
        
        # Reshape to [Total_Len*Batch, 1, Model_Dim] for Encoder
        # (This specific shape is expected by AgentFormer implementation)
        tf_in = tf_in.view(-1, 1, self.model_dim)
        
        # 3. Add Time Encoding
        # Note: We use t_offset=0 because this is the full sequence starting at 0
        tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, t_offset=0)
        # Flatten for Transformer: [Seq * Agents, 1, Dim]
        tf_in_pos_flat = tf_in_pos.view(-1, 1, self.model_dim)
        
        # 4. Create Mask
        # We need a mask that covers the full concatenated length
        # (This helper function logic is effectively what generate_mask does in agentformer.py)
        # Depending on your implementation, you might need to resize the mask to match (Seq_Len * Batch_Size)
        src_mask = agent_mask.repeat(seq_len, seq_len)
        
        # 5. Pass through AgentFormer Encoder
        feature = self.tf_encoder(tf_in_pos_flat, mask=src_mask, num_agent=agent_num)
        feature = feature.view(seq_len, batch_size, self.model_dim)
        
        # 6. GlobalPooling
        pooled_feature = torch.max(feature, dim=0)[0] # [Total_Agents, Dim]

        # 7. Classification
        score = self.out_mlp(pooled_feature)
        
        return score


if __name__ == '__main__':
    # ==========================================
    # 3. Test Script
    # ==========================================
    # Mock Config
    class Config:
        def __init__(self):
            self.motion_dim = 2
            self.forecast_dim = 2
            self.tf_model_dim = 64
            self.tf_nhead = 4
            self.tf_ff_dim = 128
            self.tf_dropout = 0.1
            self.context_encoder = {'nlayer': 2}
            self.future_decoder = {'nlayer': 2}
            self.pos_concat = True
            self.nz = 32
            self.future_frames = 12
            self.past_frames = 8
            self.nlayer = 2 
        def get(self, key, default=None): return getattr(self, key, default)

    cfg = Config()
    generator = AgentFormerGenerator(cfg)
    discriminator = AgentFormerDiscriminator(cfg)

    # --- Make Batch Data ---
    # 3 Scenes with 2, 3, and 4 agents respectively
    agents_per_scene = [2, 3, 4]
    total_agents = sum(agents_per_scene) # 9
    past_frames = 8

    # Create Pre Motion [Frames, Total_Agents, 2]
    pre_motion = torch.randn(past_frames, total_agents, 2)

    # Create Agent Mask (Block Diagonal)
    # 0 = Connected, -inf = Masked
    agent_mask = torch.full((total_agents, total_agents), float('-inf'))
    idx = 0
    for n in agents_per_scene:
        agent_mask[idx:idx+n, idx:idx+n] = 0.0
        idx += n

    data = {
        'pre_motion': pre_motion,
        'agent_num': total_agents,
        'agent_mask': agent_mask,
        # Helpers for internal code
        'pre_vel': torch.randn(past_frames, total_agents, 2),
        'pre_motion_scene_norm': pre_motion,
        'heading': torch.zeros(total_agents),
        'heading_vec': torch.zeros(total_agents, 2),
        'agent_enc_shuffle': None,
        'pre_motion_mask': torch.ones(total_agents, past_frames)
    }

    print("Running Generator...")
    pred_fake, latent_z = generator(data)

    print(f"Generator Output Shape: {pred_fake.shape}")
    print(f"Latent Vector Shape: {latent_z.shape}")

    print("Running Discriminator...")
    score = discriminator(pre_motion, pred_fake, agent_mask, total_agents)
    print(f"Discriminator Output Shape: {score.shape}")
