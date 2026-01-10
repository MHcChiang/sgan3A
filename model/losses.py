import torch
import random
import torch.nn.functional as F


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    # neg_abs = -input.abs()
    # loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    # return loss.mean()
    return F.binary_cross_entropy_with_logits(input, target)



def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.0) # 1.2
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.0) #1.2
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()

    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        # Remove .data attribute usage (deprecated in modern PyTorch)
        return torch.sum(loss) / torch.numel(loss_mask)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)



def select_best_k_scene(stack_preds, pred_real, seq_start_end, loss_mask=None):
    """
    Selects the best sample from K based on Scene-Level Sum of Squared Errors (L2).
    This preserves the joint distribution of agents in the same scene.
    
    Args:
        stack_preds: [K, Agents, Time, 2] - Generated samples
        pred_real: [Agents, Time, 2] - Ground Truth
        seq_start_end: List of tuples [(start, end), ...] - Scene boundaries
        loss_mask: [Agents, Time] or None - Valid frames mask
        
    Returns:
        best_pred: [Agents, Time, 2] - The selected best trajectories
    """
    # 1. Calculate Squared Error per Agent per Sample
    # [K, Agents, Time, 2] - [1, Agents, Time, 2]
    diff = stack_preds - pred_real.unsqueeze(0) 
    dist_sq = diff.pow(2).sum(dim=-1) # [K, Agents, Time]

    if loss_mask is not None:
        # mask: [Agents, Time] -> [1, Agents, Time]
        dist_sq = dist_sq * loss_mask.unsqueeze(0)

    # Sum over time -> Total error per agent for each K
    # error_agent: [K, Agents]
    error_agent = dist_sq.sum(dim=2) 
    
    best_pred_list = []
    scene_min_losses = []

    # 2. Scene-Level Selection
    for start, end in seq_start_end:
        # Extract errors for this scene: [K, n_agents_in_scene]
        scene_errors = error_agent[:, start:end]
        
        # Sum errors across all agents in the scene -> [K]
        # This is the "Variety Loss" metric for the scene
        scene_total_error = scene_errors.sum(dim=1)
        
        # Find the single best K for this entire scene
        min_val, best_k_idx = scene_total_error.min(dim=0)
        best_k = best_k_idx.item()

        scene_min_losses.append(min_val)
        
        # Select that sample for ALL agents in this scene
        # stack_preds[best_k]: [Agents, Time, 2] -> slice [start:end]
        best_pred_list.append(stack_preds[best_k, start:end, :, :])
        
    # Concatenate back to [Agents, Time, 2]
    best_pred = torch.cat(best_pred_list, dim=0)
    batch_l2_loss = torch.sum(torch.stack(scene_min_losses))
    return best_pred, batch_l2_loss


# 0110 Add: Hinge Loss
def g_hinge_loss(scores_fake):
    """
    Generator Hinge Loss: -E[D(G(z))]
    """
    return -torch.mean(scores_fake)

def d_hinge_loss(scores_real, scores_fake):
    """
    Discriminator Hinge Loss:
    L_D = E[max(0, 1 - D(x))] + E[max(0, 1 + D(G(z)))]
    """
    loss_real = torch.mean(F.relu(1.0 - scores_real))
    loss_fake = torch.mean(F.relu(1.0 + scores_fake))
    return loss_real + loss_fake