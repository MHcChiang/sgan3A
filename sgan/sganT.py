from models import *
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


class PoolHiddenNetT(nn.Module):
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
        super(PoolHiddenNetT, self).__init__()

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
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: list/tuple/tensor of (start_idx, end_idx) boundaries
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            num_ped = end - start
            curr_hidden = h_states.contiguous().view(-1, self.h_dim)[start:end]
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
        
        # Project from embedding_dim to h_dim to match LSTM output dimensions
        # LSTM outputs h_dim directly, but transformer outputs embedding_dim
        self.output_proj = nn.Linear(embedding_dim, h_dim)
    
    def init_hidden(self, batch):
        """
        Initialize hidden state tensors.
        
        IMPORTANT: This method is kept ONLY for compatibility with existing code.
        Transformers do NOT use hidden state initialization like LSTM does.
        
        In LSTM Encoder:
        - init_hidden() is REQUIRED to create initial hidden (h) and cell (c) states
        - These states are passed to LSTM.forward() and updated at each timestep
        - LSTM maintains memory through these states as it processes the sequence
        
        In Transformer Encoder:
        - No hidden state initialization needed
        - All timesteps are processed in parallel
        - This method is NOT called in forward() and returns unused zeros
        
        Args:
            batch: Batch size (number of trajectories in the batch)
                  - Used to determine the size of the hidden state tensors
                  
        Returns:
            Tuple (h, c) where:
            - h: Hidden state tensor of shape (num_layers, batch, h_dim)
                - Initialized to zeros (never used in Transformer)
            - c: Cell state tensor of shape (num_layers, batch, h_dim)
                - Initialized to zeros (never used in Transformer)
            - Device matches model parameters (CPU or GPU)
        """
        try:
            device = next(self.parameters()).device
        except:
            device = torch.device('cpu')
        
        # Create zero-initialized tensors matching LSTM's expected format
        # Note: These are never used in Transformer, only kept for interface compatibility
        h = torch.zeros(self.num_layers, batch, self.h_dim, device=device)
        c = torch.zeros(self.num_layers, batch, self.h_dim, device=device)
        return (h, c)

    def forward(self, obs_traj):
        """
            -obs_traj: Tensor of shape (obs_len, batch, 2)
                     - obs_len: Number of observed timesteps (e.g., 8)
                     - batch: Batch size (number of pedestrian trajectories)
                     - 2: (x, y) coordinates at each timestep
                     
        Returns:
            -final_h: Tensor of shape (num_layers, batch, h_dim)
        """
        # Encode observed trajectory
        batch = obs_traj.size(1)
        obs_len = obs_traj.size(0)
        
        # Step 1: Spatial embedding
        # Input: (obs_len, batch, 2) -> flatten to (obs_len*batch, 2)
        # Embed: (obs_len*batch, 2) -> (obs_len*batch, embedding_dim)
        # Reshape: (obs_len*batch, embedding_dim) -> (obs_len, batch, embedding_dim)
        # Note: We use obs_len explicitly instead of -1 for clarity (both work the same)
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
        
        # Step 5: Project to h_dim to match LSTM output format
        last_output = self.output_proj(last_output)  # (batch, h_dim)
        
        # Step 6: Reshape to match LSTM output format for compatibility
        final_h = last_output.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        return final_h


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
                self.pool_net = PoolHiddenNetT(
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
    
    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """
            -last_pos: Tensor of shape (batch, 2)          
            -last_pos_rel: Tensor of shape (batch, 2)    
            -state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
                       - hh: Hidden state tensor of shape (num_layers, batch, h_dim)
                         - Contains encoder's final output (used as memory)
                       - ch: Cell state tensor of shape (num_layers, batch, h_dim)
                         - Not used in Transformer
                         - In LSTM: Cell state maintains long-term memory
            -seq_start_end: A list of tuples which delimit sequences within batch [(start_idx, end_idx), ...]
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
                               
            final_decoder_h: Tensor of shape (num_layers, batch, h_dim) 
                           - Final decoder hidden state (for compatibility to LSTM)
        """
        batch = last_pos.size(0)
        
        # Extract encoder memory from state_tuple (first element)
        # state_tuple[0] shape: (num_layers, batch, h_dim)
        # We'll use the last layer as memory, convert to embedding_dim
        encoder_memory = state_tuple[0][-1]  # (batch, h_dim)
        
        # Project encoder memory to embedding_dim for transformer decoder
        # Transform to (seq_len, batch, embedding_dim) format
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
                # Need to reshape decoder_h for pooling
                decoder_h_pool = decoder_h.unsqueeze(0)  # (1, batch, h_dim)
                pool_h = self.pool_net(decoder_h_pool, seq_start_end, curr_pos)
                decoder_h = torch.cat([decoder_h, pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                # Update embedding from pooled hidden state
                decoder_h_embed = self.hidden_to_embedding(decoder_h)
            
            # Prepare next decoder input
            decoder_input = self.spatial_embedding(rel_pos).unsqueeze(0)
            
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos
        
        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        
        # Final decoder hidden state for compatibility
        # Use the last decoder_h (which may have been pooled)
        final_decoder_h = decoder_h.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, h_dim)
        
        return pred_traj_fake_rel, final_decoder_h



# test
if __name__ == "__main__":
    h_dim = 64
    encoder = Encoder_Transformer()
    decoder = Decoder_Transformer(seq_len=12, h_dim=h_dim)
    # Test the encoder
    obs_traj = torch.randn(8, 3,2 )
    final_h = encoder(obs_traj)
    print(f"✅ Encoder test passed. Output shape: {final_h.shape}")

    # Test the decoder

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

    # 3. state_tuple: (hh, ch) where each is (num_layers, batch, h_dim)
    #    hh: encoder hidden state (encoder output) - used as memory in transformer
    #    ch: cell state (not used in Transformer, but kept for compatibility)
    #    Shape: Each tensor is (1, 9, 64)
    num_layers = 1
    state_tuple = (
        torch.randn(num_layers, total_batch_size, h_dim),  # hh
        torch.randn(num_layers, total_batch_size, h_dim)   # ch
    )

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

    pred_traj_fake_rel, final_decoder_h = decoder(
        last_pos, 
        last_pos_rel, 
        state_tuple, 
        seq_start_end
    )

    print("✅ Decoder test passed!")
    print(f"  pred_traj_fake_rel: {pred_traj_fake_rel.shape} (seq_len={decoder.seq_len}, batch={total_batch_size}, 2)")
    print(f"  final_decoder_h: {final_decoder_h.shape} (num_layers={num_layers}, batch={total_batch_size}, h_dim={h_dim})")

