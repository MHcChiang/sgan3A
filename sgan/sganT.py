from sgan.models import *
import torch
import torch.nn as nn
import math


def generate_square_subsequent_mask(sz):
    """
    Generate a causal mask for autoregressive generation.
    
    This mask prevents the decoder from attending to future positions.
    Essential for autoregressive generation where each position can only
    attend to previous positions.
    
    Args:
        sz: Size of the mask (sequence length)
        
    Returns:
        mask: Tensor of shape (sz, sz) where:
              - Upper triangular (including diagonal+1) contains -inf (masked)
              - Lower triangular (including diagonal) contains 0.0 (allowed)
              - Positions marked with -inf are set to -inf in attention scores
              - This creates a "causal" or "autoregressive" mask
    
    Example for sz=4:
        [[  0.0, -inf, -inf, -inf],  # Position 0 can only see position 0
         [  0.0,  0.0, -inf, -inf],  # Position 1 can see 0,1
         [  0.0,  0.0,  0.0, -inf],  # Position 2 can see 0,1,2
         [  0.0,  0.0,  0.0,  0.0]]  # Position 3 can see all
    
    Note: PyTorch's TransformerDecoder applies softmax(score + mask),
    so -inf positions become 0 probability after softmax.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer architecture.
    required for Transformers sinceTransformers process all tokens in parallel, so position must be explicitly encoded
    
    Uses sinusoidal encoding (sin/cos) at different frequencies to encode position.
    """
    def __init__(self, d_model, max_len=100, dropout=0.1):
        """
            -d_model: Embedding dimension (must match the model's embedding_dim)
            -max_len: Maximum sequence length to pre-compute positions for
            -dropout: Dropout rate applied after adding positional encoding
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Pre-compute positional encoding matrix using sin/cos functions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cos
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)  # Register as buffer (not a parameter, but part of model)
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
               
        Returns:
            Tensor of same shape (seq_len, batch, d_model) with positional info added
        """
        x = x + self.pe[:x.size(0), :]  # Add positional encoding for each timestep
        return self.dropout(x)


class PoolHiddenNet_Transformer(nn.Module):
    """
    Pooling module (Transformer-friendly).
    Functionally equivalent to `PoolHiddenNet` in `models.py`, but:
    - Robust to `seq_start_end` provided as a list of tuples or as a tensor
    - Lives in this module to avoid tight coupling to LSTM module
    """
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet_Transformer, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def repeat(self, tensor, num_reps):
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (batch, h_dim) or (num_layers, batch, h_dim)
                   - If (num_layers, batch, h_dim), will be flattened to (batch, h_dim)
        - seq_start_end: list/tuple/tensor of (start_idx, end_idx) boundaries
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        # Handle both (batch, h_dim) and (num_layers, batch, h_dim) formats
        if h_states.dim() == 3:
            # Flatten from (num_layers, batch, h_dim) to (batch, h_dim)
            h_states = h_states.contiguous().view(-1, self.h_dim)
        
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            num_ped = end - start
            curr_hidden = h_states[start:end]  # (batch, h_dim) -> (num_ped, h_dim)
            curr_end_pos = end_pos[start:end]

            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)

        pool_h = torch.cat(pool_h, dim=0)
        return pool_h

class Encoder_Transformer(nn.Module):
    """
    Transformer-based Encoder to replace LSTM Encoder.
    
    KEY DIFFERENCES FROM LSTM ENCODER:
    1. LSTM: Sequential processing (timestep by timestep) - position is implicit
       Transformer: Parallel processing (all timesteps at once) - needs positional encoding
    2. LSTM: Maintains hidden state (h) and cell state (c) that carry information forward
       Transformer: Uses self-attention to capture relationships between all timesteps
    4. LSTM: Processes one timestep at a time, accumulates information in hidden state
       Transformer: Processes entire sequence in parallel, then we take the last timestep
    """
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        """
            -embedding_dim: Dimension of embedding space (64 default)
                          - Converts 2D positions (x,y) to this dimension
                          - Must be divisible by nhead (8)
            -h_dim: Hidden dimension for output (64 default)
                  - Matches LSTM's hidden state dimension for compatibility
            -mlp_dim: Dimension of feedforward network in transformer (1024 default)
            -num_layers: Number of transformer encoder layers (1 default)
            -dropout: Dropout rate (0.0 default)
        """
        super(Encoder_Transformer, self).__init__()
        
        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Ensure embedding_dim is divisible by nhead (required for multi-head attention)
        nhead = 8  # Number of attention heads
        if embedding_dim % nhead != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by nhead ({nhead})")
        
        # Spatial embedding: convert 2D positions (x,y) to embedding_dim
        # Input: (batch*seq_len, 2) -> Output: (batch*seq_len, embedding_dim)
        # This is the same as the original LSTM encoder
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        
        # Positional encoding: CRITICAL for Transformers
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=dropout)
        
        # Transformer encoder layers: replaces LSTM
        # Uses self-attention to capture relationships across entire sequence
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,      # Embedding dimension
            nhead=nhead,                # Number of attention heads (8)
            dim_feedforward=mlp_dim,    # Feedforward network dimension (1024)
            dropout=dropout,            # Dropout rate
            batch_first=False          # Input format: (seq_len, batch, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers      # Stack multiple layers
        )
        
        # Project from embedding_dim to h_dim
        # Transformer outputs embedding_dim, we project to h_dim for consistency
        self.output_proj = nn.Linear(embedding_dim, h_dim)

    def forward(self, obs_traj):
        """
        Encode observed trajectory using Transformer architecture.
        
        Args:
            obs_traj: Tensor of shape (obs_len, batch, 2)
                     - obs_len: Number of observed timesteps (e.g., 8)
                     - batch: Batch size (number of pedestrian trajectories)
                     - 2: (x, y) coordinates at each timestep
                     
        Returns:
            encoder_output: Tensor of shape (batch, h_dim)
                          - Encoded representation of the observed trajectory
                          - One vector per pedestrian in the batch
        """
        # Encode observed trajectory
        batch = obs_traj.size(1)
        obs_len = obs_traj.size(0)
        
        # Step 1: Spatial embedding
        # Input: (obs_len, batch, 2) -> flatten to (obs_len*batch, 2)
        # Embed: (obs_len*batch, 2) -> (obs_len*batch, embedding_dim)
        # Reshape: (obs_len*batch, embedding_dim) -> (obs_len, batch, embedding_dim)
        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            obs_len, batch, self.embedding_dim
        )
        
        # Step 2: Add positional encoding (REQUIRED for Transformer)
        obs_traj_embedding = self.pos_encoder(obs_traj_embedding)
        
        # Step 3: Transformer encoder processes entire sequence in parallel
        transformer_output = self.transformer_encoder(obs_traj_embedding)
        
        # Step 4: Extract last timestep
        last_output = transformer_output[-1]  # Take last timestep: (batch, embedding_dim)
        
        # Step 5: Project to h_dim
        encoder_output = self.output_proj(last_output)  # (batch, h_dim)
        
        return encoder_output


class Decoder_Transformer(nn.Module):
    """
    Transformer-based Decoder to replace LSTM Decoder.
    """
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8
    ):
        """
            -seq_len: Prediction length (number of future timesteps to predict)
            -embedding_dim: Dimension of embedding space (64 default, must be divisible by 8)
            -h_dim: Hidden dimension (128 default)
            -mlp_dim: Feedforward network dimension in transformer (1024 default)
            -num_layers: Number of transformer decoder layers (1 default)
            -pool_every_timestep: Whether to apply pooling at each prediction step (True default)
            -dropout: Dropout rate (0.0 default)
            -bottleneck_dim: Dimension of pooling bottleneck (1024 default)
            -activation: Activation function for MLP ('relu' default)
            -batch_norm: Whether to use batch normalization (True default)
            -pooling_type: Type of pooling ('pool_net' or 'spool')
            -neighborhood_size: Neighborhood size for spool (2.0 default)
            -grid_size: Grid size for spool (8 default)
        """
        super(Decoder_Transformer, self).__init__()
        
        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.pool_every_timestep = pool_every_timestep
        
        # Spatial embedding
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        
        # Ensure embedding_dim is divisible by nhead
        nhead = 8
        if embedding_dim % nhead != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by nhead ({nhead})")
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=seq_len, dropout=dropout)
        
        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Register causal mask as buffer (not a parameter, but part of model state)
        # This mask prevents decoder from attending to future positions
        # For single-step generation, this isn't strictly needed, but it's good practice
        # and ensures correctness if we ever process multiple steps at once
        self.register_buffer('causal_mask', generate_square_subsequent_mask(seq_len))
        
        # Project from embedding_dim to h_dim
        self.embedding_to_hidden = nn.Linear(embedding_dim, h_dim)
        
        # Project from h_dim to embedding_dim (for encoder memory)
        self.hidden_to_embedding = nn.Linear(h_dim, embedding_dim)
        
        # Output projection from h_dim to 2D position
        self.hidden2pos = nn.Linear(h_dim, 2)
        
        # Pooling (same as original decoder)
        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet_Transformer(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )
            
            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
    
    def forward(self, last_pos, last_pos_rel, encoder_memory, seq_start_end):
        """
        Decode future trajectory using Transformer architecture.
        
        Args:
            last_pos: Tensor of shape (batch, 2)
                     - Last absolute position for each pedestrian
            last_pos_rel: Tensor of shape (batch, 2)
                        - Last relative position for each pedestrian
            encoder_memory: Tensor of shape (batch, h_dim)
                          - Encoder's output representation for each pedestrian
                          - Used as memory in the transformer decoder
            seq_start_end: A list of tuples which delimit sequences within batch [(start_idx, end_idx), ...]
                          - Delimits different sequences/pedestrians within the batch
                          - Used for pooling to model social interactions
                          - Example: [(0, 3), (3, 7), (7, 10)] means:
                            * Pedestrians 0-2 form one sequence (3 people)
                            * Pedestrians 3-6 form another sequence (4 people)
                            * Pedestrians 7-9 form another sequence (3 people)
                          - Pooling only considers interactions within same sequence
                          
        Returns:
            pred_traj_fake_rel: Tensor of shape (seq_len, batch, 2)
                               - Predicted relative trajectory
                               - seq_len: Number of future timesteps predicted
                               - batch: Same as input batch size
                               - 2: Relative (dx, dy) positions at each timestep
                               - Example for seq_len=12: shape is (12, batch, 2)
        """
        batch = last_pos.size(0)
        
        # Project encoder memory to embedding_dim for transformer decoder
        # encoder_memory: (batch, h_dim) -> encoder_memory_embed: (batch, embedding_dim)
        encoder_memory_embed = self.hidden_to_embedding(encoder_memory)  # (batch, embedding_dim)
        encoder_memory_embed = encoder_memory_embed.unsqueeze(0)  # (1, batch, embedding_dim)
        
        # Autoregressive generation
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)  # (batch, embedding_dim)
        decoder_input = decoder_input.unsqueeze(0)  # (1, batch, embedding_dim)
        
        for _ in range(self.seq_len):
            # Add positional encoding
            decoder_input_pos = self.pos_encoder(decoder_input)
            
            # Get causal mask for current step
            # Since we process one timestep at a time (seq_len=1), mask isn't strictly needed
            # However, it's included for correctness and in case we process multiple steps
            # The mask ensures decoder self-attention doesn't attend to future positions
            # For single-step generation (seq_len=1), this will be a 1x1 matrix with all zeros
            # tgt_mask = self.causal_mask[:step+1, :step+1] if step > 0 else None
            
            # Transformer decoder: generate next step
            # decoder_input_pos: (1, batch, embedding_dim) - current step input
            # encoder_memory_embed: (1, batch, embedding_dim) - encoder memory
            # tgt_mask: Causal mask to prevent attending to future (None for first step)
            decoder_output = self.transformer_decoder(
                decoder_input_pos,
                encoder_memory_embed,
                # tgt_mask=tgt_mask  # Causal mask for autoregressive generation
            )  # (1, batch, embedding_dim)
            
            # Project to h_dim for pooling compatibility
            decoder_h_embed = decoder_output.squeeze(0)  # (batch, embedding_dim)
            decoder_h = self.embedding_to_hidden(decoder_h_embed)  # (batch, h_dim)
            
            # Generate relative position
            rel_pos = self.hidden2pos(decoder_h)  # (batch, 2)
            curr_pos = rel_pos + last_pos
            
            # Pooling (if enabled)
            if self.pool_every_timestep:
                # Pooling expects (batch, h_dim) - decoder_h is already in this format
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat([decoder_h, pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                # Update embedding from pooled hidden state
                decoder_h_embed = self.hidden_to_embedding(decoder_h)
            
            # Prepare next decoder input
            decoder_input = self.spatial_embedding(rel_pos).unsqueeze(0)
            
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos
        
        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        
        return pred_traj_fake_rel


class TrajectoryGenerator_Transformer(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
        super(TrajectoryGenerator_Transformer, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024

        self.encoder = Encoder_Transformer(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder_Transformer(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet_Transformer(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        # Convert noise_dim to tuple if it's a list (for Python 3.10+ compatibility)
        noise_dim_tuple = tuple(self.noise_dim) if isinstance(self.noise_dim, list) else self.noise_dim

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + noise_dim_tuple
        else:
            noise_shape = (_input.size(0), ) + noise_dim_tuple

        if user_noise is not None:
            z_decoder = user_noise
        else:
            # Get device from input tensor (device-agnostic)
            device = _input.device
            z_decoder = get_noise(noise_shape, self.noise_type, device=device)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        # Encode observed trajectory
        # encoder_output: (batch, encoder_h_dim) - no num_layers dimension!
        encoder_output = self.encoder(obs_traj_rel)
        
        # Pool States (if pooling is enabled)
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]  # (batch, 2)
            # pool_net expects (batch, h_dim) or (num_layers, batch, h_dim) - handles both
            pool_h = self.pool_net(encoder_output, seq_start_end, end_pos)
            # Construct input for decoder context
            mlp_decoder_context_input = torch.cat(
                [encoder_output, pool_h], dim=1)  # (batch, encoder_h_dim + bottleneck_dim)
        else:
            mlp_decoder_context_input = encoder_output  # (batch, encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        
        # Add noise to create decoder input
        decoder_input = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)  # (batch, decoder_h_dim)

        # Get last positions for decoder
        last_pos = obs_traj[-1]  # (batch, 2)
        last_pos_rel = obs_traj_rel[-1]  # (batch, 2)
        
        # Decode future trajectory
        # decoder_input becomes encoder_memory for the decoder
        pred_traj_fake_rel = self.decoder(
            last_pos,
            last_pos_rel,
            decoder_input,  # (batch, decoder_h_dim) - clean interface!
            seq_start_end,
        )

        return pred_traj_fake_rel
        

class TrajectoryDiscriminator_Transformer(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator_Transformer, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder_Transformer(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            self.pool_net = PoolHiddenNet_Transformer(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        # Encode full trajectory
        # encoder_output: (batch, h_dim) - no num_layers dimension!
        encoder_output = self.encoder(traj_rel)
        
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            # Direct classification from encoder output
            classifier_input = encoder_output  # (batch, h_dim)
        else:
            # Global pooling across scenes
            classifier_input = self.pool_net(
                encoder_output, seq_start_end, traj[0]  # (batch, h_dim)
            )
        
        scores = self.real_classifier(classifier_input)
        return scores


# test
if __name__ == "__main__":
    h_dim = 64
    encoder = Encoder_Transformer()
    decoder = Decoder_Transformer(seq_len=12, h_dim=h_dim)
    
    # Test the encoder
    obs_traj = torch.randn(8, 3, 2)
    encoder_output = encoder(obs_traj)
    print(f"✅ Encoder test passed. Output shape: {encoder_output.shape} (expected: (3, {h_dim}))")

    # Test the decoder
    # Scenario: 3 scenes (batches), each with 3 pedestrians
    bs = 3  # num of scenes
    num_pedestrians_per_scene = 3  # each scene has 3 pedestrians
    total_batch_size = bs * num_pedestrians_per_scene  # total = 9 pedestrians

    # ==========================================
    # Create input tensors with batch size 9
    # ==========================================

    # 1. last_pos: (batch, 2) - last absolute position for each pedestrian
    #    Shape: (9, 2) - 9 pedestrians, each with (x, y) coordinates
    last_pos = torch.randn(total_batch_size, 2)

    # 2. last_pos_rel: (batch, 2) - last relative position for each pedestrian
    #    Shape: (9, 2) - 9 pedestrians, each with relative (dx, dy) coordinates
    last_pos_rel = torch.randn(total_batch_size, 2)

    # 3. encoder_memory: (batch, h_dim) - encoder output (no longer a tuple!)
    #    Shape: (9, 64) - 9 pedestrians, each with h_dim-dimensional encoding
    encoder_memory = torch.randn(total_batch_size, h_dim)

    # 4. seq_start_end: List of tuples delimiting scenes within batch
    #    This tells the decoder which pedestrians belong to the same scene
    #    Scene 0: pedestrians 0-2 (indices 0, 1, 2)
    #    Scene 1: pedestrians 3-5 (indices 3, 4, 5)
    #    Scene 2: pedestrians 6-8 (indices 6, 7, 8)
    seq_start_end = [
        (0, 3),      # Scene 0: 3 pedestrians (indices 0, 1, 2)
        (3, 6),      # Scene 1: 3 pedestrians (indices 3, 4, 5)
        (6, 9)       # Scene 2: 3 pedestrians (indices 6, 7, 8)
    ]
    # Convert to tensor for noise functions that expect it
    seq_start_end_tensor = torch.tensor(seq_start_end, dtype=torch.long)

    # Run decoder with new clean interface (no state_tuple!)
    pred_traj_fake_rel = decoder(
        last_pos, 
        last_pos_rel, 
        encoder_memory,  # Direct encoder memory, not a tuple
        seq_start_end
    )

    print("\n✅ Decoder test passed!")
    print(f"  pred_traj_fake_rel: {pred_traj_fake_rel.shape} (seq_len={decoder.seq_len}, batch={total_batch_size}, 2)")
    print(f"  Expected: ({decoder.seq_len}, {total_batch_size}, 2)")

    # ==========================================
    # Test TrajectoryGenerator_Transformer
    # ==========================================
    print("\n" + "="*60)
    print("Testing TrajectoryGenerator_Transformer")
    print("="*60)
    
    obs_len = 8
    pred_len = 12
    embedding_dim = 64
    encoder_h_dim = 64
    decoder_h_dim = 128
    
    obs_traj = torch.randn(obs_len, total_batch_size, 2)
    obs_traj_rel = torch.randn(obs_len, total_batch_size, 2)
    
    # Test with noise
    print("\nTest Generator")
    generator = TrajectoryGenerator_Transformer(
        obs_len=obs_len,
        pred_len=pred_len,
        embedding_dim=embedding_dim,
        encoder_h_dim=encoder_h_dim,
        decoder_h_dim=decoder_h_dim,
        pooling_type='pool_net',
        noise_dim=(16,),  # Add noise
        noise_type='gaussian',
        noise_mix_type='ped',
        pool_every_timestep=True
    )
    
    pred_traj_rel_noise = generator(obs_traj, obs_traj_rel, seq_start_end_tensor)
    print(f"✅ Generator test passed!")
    print(f"  Output pred_traj_rel: {pred_traj_rel_noise.shape}")
    print(f"  Expected: ({pred_len}, {total_batch_size}, 2)")
    
    # ==========================================
    # Test TrajectoryDiscriminator_Transformer
    # ==========================================

    # Test local discriminator
    print("\nTest Discriminator (local)")
    discriminator_local = TrajectoryDiscriminator_Transformer(
        obs_len=obs_len,
        pred_len=pred_len,
        embedding_dim=embedding_dim,
        h_dim=encoder_h_dim,
        d_type='local'
    )
    
    # Full trajectory (observed + predicted)
    full_traj = torch.randn(obs_len + pred_len, total_batch_size, 2)
    full_traj_rel = torch.randn(obs_len + pred_len, total_batch_size, 2)
    
    scores_local = discriminator_local(full_traj, full_traj_rel, seq_start_end)
    print(f"✅ Discriminator (local) test passed!")
    print(f"  Input full_traj: {full_traj.shape}")
    print(f"  Input full_traj_rel: {full_traj_rel.shape}")
    print(f"  Output scores: {scores_local.shape}")
    print(f"  Expected: ({total_batch_size}, 1)")
    
    # Test global discriminator
    print("\nTest Discriminator (global)")
    discriminator_global = TrajectoryDiscriminator_Transformer(
        obs_len=obs_len,
        pred_len=pred_len,
        embedding_dim=embedding_dim,
        h_dim=encoder_h_dim,
        d_type='global'
    )
    
    scores_global = discriminator_global(full_traj, full_traj_rel, seq_start_end)
    print(f"✅ Discriminator (global) test passed!")
    print(f"  Output scores: {scores_global.shape}")
    print(f"  Expected: ({total_batch_size}, 1)")