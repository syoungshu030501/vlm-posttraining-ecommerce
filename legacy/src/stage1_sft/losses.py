"""
Auxiliary losses for Stage 1 SFT:
  - SupCon (Supervised Contrastive Loss) for violation-aware alignment
  - Hallucination triplet loss for grounding attribute references
"""
import torch
import torch.nn.functional as F


def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss.

    Args:
        embeddings: (B, D) - [EOS] token hidden states from last layer, NOT yet normalized
        labels:     (B,)   - binary violation labels (0=compliant, 1=violating)
        temperature: scalar

    Returns:
        Scalar loss.
    """
    embeddings = F.normalize(embeddings, dim=-1)  # (B, D)
    sim = torch.matmul(embeddings, embeddings.T) / temperature  # (B, B)

    # Positive mask: same label, exclude diagonal
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0.0)

    # Log-sum-exp trick for numerical stability
    exp_sim = torch.exp(sim - sim.detach().max(dim=1, keepdim=True).values)
    log_prob = sim - sim.detach().max(dim=1, keepdim=True).values \
               - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    n_pos = pos_mask.sum(dim=1).clamp(min=1.0)
    mean_log_pos = (log_prob * pos_mask).sum(dim=1) / n_pos
    return -mean_log_pos.mean()


def hallucination_triplet_loss(
    img_embed: torch.Tensor,
    pos_attr_embed: torch.Tensor,
    neg_attr_embed: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Triplet loss to suppress hallucination.

    anchor  = image embedding
    positive = embedding of real attribute description
    negative = embedding of hallucinated attribute description

    Args:
        img_embed:      (B, D)
        pos_attr_embed: (B, D)
        neg_attr_embed: (B, D)
        margin:         triplet margin

    Returns:
        Scalar loss.
    """
    pos_sim = F.cosine_similarity(img_embed, pos_attr_embed)
    neg_sim = F.cosine_similarity(img_embed, neg_attr_embed)
    return F.relu(neg_sim - pos_sim + margin).mean()
