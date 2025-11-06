# Removing LSTM Compatibility from Transformer Components

## Overview

Currently, the Transformer encoder/decoder maintain LSTM compatibility by:
- Returning/accepting `(num_layers, batch, h_dim)` format
- Using `(hh, ch)` tuple format for state
- Maintaining unused `num_layers` dimension

If we remove LSTM compatibility, we can simplify the interface significantly.

---

## Current State (LSTM-Compatible)

### Encoder_Transformer
- **Returns**: `(num_layers, batch, h_dim)` - but only `num_layers=1` is meaningful
- **Has**: `init_hidden()` method (never used, kept for compatibility)

### Decoder_Transformer
- **Accepts**: `state_tuple = (hh, ch)` where:
  - `hh`: `(num_layers, batch, h_dim)` - only uses `hh[-1]` to get `(batch, h_dim)`
  - `ch`: `(num_layers, batch, h_dim)` - **completely ignored**
- **Returns**: `(pred_traj_fake_rel, final_decoder_h)` where `final_decoder_h` is `(num_layers, batch, h_dim)` - **not used by TrajectoryGenerator**

### TrajectoryGenerator
- Creates fake `decoder_c` tensor just for compatibility
- Passes `(decoder_h, decoder_c)` to decoder
- Ignores returned `final_decoder_h`

### TrajectoryDiscriminator
- Uses encoder, calls `.squeeze()` on output
- **No changes needed** - will work with either format

---

## Proposed Changes (No LSTM Compatibility)

### 1. Encoder_Transformer Changes

**Current:**
```python
def forward(self, obs_traj):
    # ... processing ...
    final_h = last_output.unsqueeze(0).repeat(self.num_layers, 1, 1)
    return final_h  # (num_layers, batch, h_dim)
```

**New:**
```python
def forward(self, obs_traj):
    # ... processing ...
    return last_output  # (batch, h_dim) - directly return
```

**Remove:**
- `init_hidden()` method (no longer needed)
- `num_layers` dimension from output

### 2. Decoder_Transformer Changes

**Current:**
```python
def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
    # Extract from tuple: encoder_memory = state_tuple[0][-1]  # (batch, h_dim)
    # ... processing ...
    return pred_traj_fake_rel, final_decoder_h  # final_decoder_h not used
```

**New:**
```python
def forward(self, last_pos, last_pos_rel, encoder_memory, seq_start_end):
    # encoder_memory: (batch, h_dim) - directly passed
    # ... processing ...
    return pred_traj_fake_rel  # Only return trajectory
```

**Remove:**
- `state_tuple` parameter - replace with `encoder_memory: (batch, h_dim)`
- `final_decoder_h` return value
- `num_layers` handling

### 3. TrajectoryGenerator Changes

**Current (lines 559-598):**
```python
final_encoder_h = self.encoder(obs_traj_rel)  # (num_layers, batch, h_dim)

# ... pooling and noise addition ...

decoder_h = self.add_noise(...)  # (batch, decoder_h_dim)
decoder_h = torch.unsqueeze(decoder_h, 0)  # (1, batch, decoder_h_dim)

decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim, device=device)
state_tuple = (decoder_h, decoder_c)

decoder_out = self.decoder(last_pos, last_pos_rel, state_tuple, seq_start_end)
pred_traj_fake_rel, final_decoder_h = decoder_out
```

**New:**
```python
encoder_memory = self.encoder(obs_traj_rel)  # (batch, encoder_h_dim)

# ... pooling and noise addition ...

decoder_input = self.add_noise(...)  # (batch, decoder_h_dim)

# Pass encoder memory directly (no tuple needed)
pred_traj_fake_rel = self.decoder(
    last_pos, 
    last_pos_rel, 
    decoder_input,  # This becomes the encoder memory for decoder
    seq_start_end
)
```

**Note:** The decoder input needs to be the processed encoder output (with pooling/noise), so we'd pass `decoder_input` as the encoder memory.

---

## Impact Analysis

### ✅ TrajectoryGenerator
**Requires changes:**
- Update encoder output handling (remove `num_layers` dimension)
- Remove `decoder_c` creation
- Update decoder call signature
- Remove `final_decoder_h` unpacking

**Complexity:** Medium - but makes code cleaner

### ✅ TrajectoryDiscriminator
**No changes needed:**
- Currently calls `.squeeze()` on encoder output
- Will work with `(batch, h_dim)` directly
- No dependency on decoder

**Complexity:** None

### ✅ Pooling Components
**No changes needed:**
- `PoolHiddenNet` expects `(num_layers, batch, h_dim)` but can handle `(batch, h_dim)` with `.view()`
- Or we can update pooling to accept `(batch, h_dim)` directly

**Complexity:** Low (minor update if needed)

---

## Example: Clean Transformer Interface

### Encoder_Transformer (Simplified)
```python
class Encoder_Transformer(nn.Module):
    def forward(self, obs_traj):
        # ... processing ...
        last_output = transformer_output[-1]  # (batch, embedding_dim)
        final_h = self.output_proj(last_output)  # (batch, h_dim)
        return final_h  # Simple: (batch, h_dim)
```

### Decoder_Transformer (Simplified)
```python
class Decoder_Transformer(nn.Module):
    def forward(self, last_pos, last_pos_rel, encoder_memory, seq_start_end):
        # encoder_memory: (batch, h_dim) - directly use
        encoder_memory_embed = self.hidden_to_embedding(encoder_memory)
        # ... rest of processing ...
        return pred_traj_fake_rel  # Only return trajectory
```

### TrajectoryGenerator (Simplified)
```python
def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
    # Encode
    encoder_memory = self.encoder(obs_traj_rel)  # (batch, encoder_h_dim)
    
    # Pool (if needed)
    if self.pooling_type:
        pool_h = self.pool_net(encoder_memory, seq_start_end, end_pos)
        context = torch.cat([encoder_memory, pool_h], dim=1)
    else:
        context = encoder_memory
    
    # Add noise
    decoder_input = self.add_noise(context, seq_start_end, user_noise)
    
    # Decode (no tuple, no fake cell state)
    pred_traj_fake_rel = self.decoder(
        last_pos, 
        last_pos_rel, 
        decoder_input,  # Clean interface
        seq_start_end
    )
    
    return pred_traj_fake_rel
```

---

## Benefits of Removing LSTM Compatibility

1. **Cleaner Interface**: No fake `num_layers` dimension or unused `ch` state
2. **Less Confusion**: Clear that Transformers don't use hidden/cell states
3. **Better Performance**: Slightly less memory allocation (no fake tensors)
4. **Easier to Understand**: Code matches Transformer architecture directly
5. **Type Safety**: Clearer tensor shapes in function signatures

## Migration Path

1. Update `Encoder_Transformer.forward()` to return `(batch, h_dim)`
2. Remove `Encoder_Transformer.init_hidden()` method
3. Update `Decoder_Transformer.forward()` signature and implementation
4. Update `TrajectoryGenerator.forward()` to use new interface
5. Test with existing training/evaluation scripts
6. Update any other code that uses encoder/decoder directly

---

## Summary

**Will this influence TrajectoryGenerator?** 
- ✅ Yes - requires updates to handle new encoder/decoder interface
- Changes are straightforward and make the code cleaner

**Will this influence Discriminator?**
- ❌ No - Discriminator only uses encoder, and it already calls `.squeeze()` which works with new format

**Overall Impact:**
- Low risk, high benefit
- Makes Transformer architecture more explicit
- Removes unnecessary compatibility code

