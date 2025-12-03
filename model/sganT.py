import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from model.agentformer_lib import (
        AgentFormerEncoderLayer,
        AgentFormerDecoderLayer,
        AgentFormerEncoder,
        AgentFormerDecoder,
    )
except ImportError:
    from agentformer_lib import (
        AgentFormerEncoderLayer,
        AgentFormerDecoderLayer,
        AgentFormerEncoder,
        AgentFormerDecoder,
    )


def _normalize_seq_start_end(
    seq_start_end: Sequence[Tuple[int, int]] | torch.Tensor | None,
    batch_size: int,
) -> List[Tuple[int, int]]:
    """
    Convert the seq_start_end structure used by SGAN data loader into a plain
    python list of (start, end) tuples. When the loader does not provide this
    information we treat the whole batch as a single scene.
    """
    if seq_start_end is None:
        return [(0, batch_size)]
    if isinstance(seq_start_end, torch.Tensor):
        seq_start_end = seq_start_end.tolist()
    return [(int(start), int(end)) for start, end in seq_start_end]


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


class TemporalAgentEncoding(nn.Module):
    """
    Lightweight version of AgentFormer's positional + agent encoding.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self._cache: dict[int, torch.Tensor] = {}

    def _sinusoidal_encoding(self, length: int, device: torch.device) -> torch.Tensor:
        if length in self._cache:
            return self._cache[length].to(device)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(length, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self._cache[length] = pe
        return pe.to(device)

    def forward(self, tokens: torch.Tensor, agent_num: int) -> torch.Tensor:
        total = tokens.size(0)
        if total % agent_num != 0:
            raise ValueError('Token length must be divisible by number of agents.')
        steps = total // agent_num
        device = tokens.device
        time_enc = (
            self._sinusoidal_encoding(steps, device)
            .repeat_interleave(agent_num, dim=0)
            .unsqueeze(1)
        )
        agent_enc = (
            self._sinusoidal_encoding(agent_num, device)
            .repeat(steps, 1)
            .unsqueeze(1)
        )
        tokens = tokens + time_enc + agent_enc
        return self.dropout(tokens)


class AgentFormerSceneEncoder(nn.Module):
    """
    Encodes a batch of trajectories scene-by-scene using AgentFormer's encoder.
    Uses absolute positions (like AgentFormer), not relative trajectories.
    """

    def __init__(
        self,
        obs_len: int,
        embedding_dim: int,
        h_dim: int,
        mlp_dim: int,
        num_layers: int,
        dropout: float,
        nhead: int = 8,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.embed = nn.Linear(2, embedding_dim)
        self.temporal_encoding = TemporalAgentEncoding(embedding_dim, dropout)
        tf_cfg = {'gaussian_kernel': False, 'sep_attn': True}
        encoder_layer = AgentFormerEncoderLayer(
            tf_cfg,
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=mlp_dim,
            dropout=dropout,
        )
        self.encoder = AgentFormerEncoder(encoder_layer, num_layers)
        self.latent_proj = nn.Linear(embedding_dim, h_dim)

    def forward(
        self,
        obs_traj: torch.Tensor,
        seq_start_end: Sequence[Tuple[int, int]] | torch.Tensor | None,
    ):
        batch = obs_traj.size(1)
        device = obs_traj.device
        splits = _normalize_seq_start_end(seq_start_end, batch)
        agent_latents = torch.zeros(batch, self.h_dim, device=device)
        memories = []
        for start, end in splits:
            agent_num = end - start
            if agent_num == 0:
                continue
            scene_traj = obs_traj[:, start:end]  # (T, agent_num, 2) - absolute positions
            tokens = self.embed(scene_traj.reshape(-1, 2)).view(-1, 1, self.embedding_dim)
            tokens = self.temporal_encoding(tokens, agent_num)
            encoded = self.encoder(tokens, num_agent=agent_num)
            encoded_seq = encoded.view(-1, agent_num, self.embedding_dim)
            agent_latents[start:end] = self.latent_proj(encoded_seq[-1])
            memories.append(
                {
                    'slice': (start, end),
                    'agent_num': agent_num,
                    'memory': encoded,
                    'agent_mask': torch.zeros(agent_num, agent_num, device=device),
                }
            )
        return agent_latents, memories


class AgentFormerSceneDecoder(nn.Module):
    """
    Autoregressive decoder that mirrors AgentFormer's future decoder but is
    tailored to the SGAN generator interface.
    """

    def __init__(
        self,
        pred_len: int,
        embedding_dim: int,
        decoder_h_dim: int,
        mlp_dim: int,
        num_layers: int,
        dropout: float,
        nhead: int = 8,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.temporal_encoding = TemporalAgentEncoding(embedding_dim, dropout)
        self.input_fc = nn.Linear(2 + decoder_h_dim, embedding_dim)
        tf_cfg = {'gaussian_kernel': False, 'sep_attn': True}
        decoder_layer = AgentFormerDecoderLayer(
            tf_cfg,
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=mlp_dim,
            dropout=dropout,
        )
        self.decoder = AgentFormerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(embedding_dim, 2)

    def forward(
        self,
        last_pos: torch.Tensor,
        decoder_state: torch.Tensor,
        seq_start_end: Sequence[Tuple[int, int]] | torch.Tensor | None,
        memories: List[dict],
    ) -> torch.Tensor:
        batch = last_pos.size(0)
        device = last_pos.device
        splits = _normalize_seq_start_end(seq_start_end, batch)
        pred = torch.zeros(self.pred_len, batch, 2, device=device)
        mem_dict = {(entry['slice'][0], entry['slice'][1]): entry for entry in memories}
        for start, end in splits:
            agent_num = end - start
            if agent_num == 0:
                continue
            scene_memory = mem_dict[(start, end)]
            memory = scene_memory['memory']
            agent_mask = scene_memory['agent_mask']
            scene_pos = last_pos[start:end]
            scene_state = decoder_state[start:end]
            decoder_tokens = []
            scene_outputs = []
            for _ in range(self.pred_len):
                # Use absolute positions as input (like AgentFormer)
                step_feat = torch.cat([scene_pos, scene_state], dim=-1)
                step_embed = self.input_fc(step_feat).unsqueeze(1)
                decoder_tokens.append(step_embed)
                tgt = torch.cat(decoder_tokens, dim=0)
                tgt = self.temporal_encoding(tgt, agent_num)
                mem_mask = generate_mask(tgt.size(0), memory.size(0), agent_num, agent_mask)
                tgt_mask = generate_ar_mask(tgt.size(0), agent_num, agent_mask)
                output, _ = self.decoder(
                    tgt,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=mem_mask,
                    num_agent=agent_num,
                )
                latest = output[-agent_num:].squeeze(1)
                rel_step = self.output_proj(latest)  # Output relative displacement
                scene_outputs.append(rel_step)
                scene_pos = scene_pos + rel_step  # Update absolute position
            scene_pred = torch.stack(scene_outputs, dim=0)
            pred[:, start:end, :] = scene_pred
        return pred


def build_mlp(dim_list, activation='relu', batch_norm=False, dropout=0.0):
    layers = []
    for i in range(len(dim_list) - 1):
        layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_list[i + 1]))
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TrajectoryGenerator_AgentFormer(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        embedding_dim=64,
        encoder_h_dim=64,
        decoder_h_dim=128,
        mlp_dim=1024,
        num_layers=1,
        noise_dim=(0,),
        noise_type='gaussian',
        noise_mix_type='ped',
        pooling_type=None,
        pool_every_timestep=True,
        dropout=0.0,
        bottleneck_dim=1024,
        activation='relu',
        batch_norm=True,
        neighborhood_size=2.0,
        grid_size=8,
    ):
        super().__init__()
        del pooling_type, pool_every_timestep, bottleneck_dim, activation, batch_norm
        del neighborhood_size, grid_size

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.noise_dim = None if not noise_dim or noise_dim[0] == 0 else noise_dim
        self.noise_first_dim = 0 if self.noise_dim is None else noise_dim[0]
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.encoder = AgentFormerSceneEncoder(
            obs_len=obs_len,
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = AgentFormerSceneDecoder(
            pred_len=pred_len,
            embedding_dim=embedding_dim,
            decoder_h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        input_dim = encoder_h_dim
        self.mlp_decoder_context = None
        if self.noise_dim or encoder_h_dim != decoder_h_dim:
            mlp_decoder_context_dims = [
                input_dim,
                mlp_dim,
                decoder_h_dim - self.noise_first_dim,
            ]
            self.mlp_decoder_context = build_mlp(mlp_decoder_context_dims)

    def add_noise(self, _input, seq_start_end, user_noise=None):
        if not self.noise_dim:
            return _input
        if isinstance(self.noise_dim, list):
            noise_dim_tuple = tuple(self.noise_dim)
        else:
            noise_dim_tuple = self.noise_dim
        if self.noise_mix_type == 'global':
            seq = torch.as_tensor(seq_start_end, device=_input.device)
            noise_shape = (seq.size(0),) + noise_dim_tuple
        else:
            noise_shape = (_input.size(0),) + noise_dim_tuple
        if user_noise is not None:
            z = user_noise
        else:
            z = get_noise(noise_shape, self.noise_type, device=_input.device)
        if self.noise_mix_type == 'global':
            seq = torch.as_tensor(seq_start_end, device=_input.device)
            pieces = []
            for idx, (start, end) in enumerate(seq.tolist()):
                vec = z[idx].view(1, -1).repeat(end - start, 1)
                pieces.append(torch.cat([_input[start:end], vec], dim=1))
            return torch.cat(pieces, dim=0)
        return torch.cat([_input, z], dim=1)

    def forward(self, obs_traj, seq_start_end, user_noise=None):
        # Use absolute positions (like AgentFormer)
        encoder_output, memories = self.encoder(obs_traj, seq_start_end)
        if self.mlp_decoder_context is not None:
            decoder_context = self.mlp_decoder_context(encoder_output)
        else:
            decoder_context = encoder_output
        decoder_state = self.add_noise(decoder_context, seq_start_end, user_noise)
        last_pos = obs_traj[-1]
        return self.decoder(
            last_pos,
            decoder_state,
            seq_start_end,
            memories,
        )

    @classmethod
    def from_cfg(cls, cfg: dict):
        """Factory helper that accepts a config dictionary."""
        return cls(**cfg)


class TrajectoryDiscriminator_AgentFormer(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=1024,
        num_layers=1,
        activation='relu',
        batch_norm=True,
        dropout=0.0,
        d_type='local',
    ):
        super().__init__()
        self.d_type = d_type
        self.encoder = AgentFormerSceneEncoder(
            obs_len=obs_len + pred_len,
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        classifier_dims = [h_dim, mlp_dim, mlp_dim]
        self.classifier = build_mlp(
            classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        self.classifier.add_module('final_output', nn.Linear(mlp_dim, 1))

    def forward(self, traj, seq_start_end=None):
        # Use absolute positions (like AgentFormer)
        enc, _ = self.encoder(traj, seq_start_end)
        if self.d_type == 'global' and seq_start_end is not None:
            splits = _normalize_seq_start_end(seq_start_end, enc.size(0))
            pooled = []
            for start, end in splits:
                pooled_vec = enc[start:end].mean(dim=0, keepdim=True)
                pooled.append(pooled_vec.repeat(end - start, 1))
            enc = torch.cat(pooled, dim=0)
        scores = self.classifier(enc)
        return scores

    @classmethod
    def from_cfg(cls, cfg: dict):
        return cls(**cfg)


if __name__ == '__main__':
    torch.manual_seed(0)
    obs_len, pred_len = 8, 12
    batch = 6
    seq_start_end = torch.tensor([[0, 3], [3, 6]])
    obs_traj = torch.randn(obs_len, batch, 2)
    generator = TrajectoryGenerator_AgentFormer(
        obs_len=obs_len,
        pred_len=pred_len,
        embedding_dim=64,
        encoder_h_dim=64,
        decoder_h_dim=128,
        mlp_dim=256,
        num_layers=2,
        noise_dim=(16,),
    )
    pred = generator(obs_traj, seq_start_end)
    print('Generator output', pred.shape)
    discriminator = TrajectoryDiscriminator_AgentFormer(
        obs_len=obs_len,
        pred_len=pred_len,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=256,
        num_layers=2,
    )
    traj = torch.randn(obs_len + pred_len, batch, 2)
    scores = discriminator(traj, seq_start_end)
    print('Discriminator output', scores.shape)


# Backwards-compatible aliases for training scripts that still import the old
# Transformer naming.
TrajectoryGenerator_Transformer = TrajectoryGenerator_AgentFormer
TrajectoryDiscriminator_Transformer = TrajectoryDiscriminator_AgentFormer
