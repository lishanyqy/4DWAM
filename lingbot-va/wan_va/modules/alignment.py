import math

import torch
import torch.nn.functional as F


def motion_incremental_alignment(
    a,
    b,
    Tokens=192,
    pool="mean",
    normalize_before_delta=False,
    normalize_after_delta=True,
    eps=1e-8,
):
    """
    Motion incremental alignment via pooled temporal delta.

    Args:
        a: Tensor, shape (B, F*Tokens, D)
        b: Tensor, shape (B, F*Tokens, D)
        Tokens: int, number of tokens per frame/chunk
        pool: "mean" or "max"
        normalize_before_delta: whether to normalize pooled frame features before differencing
        normalize_after_delta: whether to normalize delta features before cosine loss
        eps: numerical stability

    Returns:
        loss: scalar
    """
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} vs {b.shape}")

    if a.dim() != 3:
        raise ValueError(f"Expected a and b to be 3D tensors of shape (B, F*Tokens, D), got {a.shape}")

    B, L, D = a.shape

    if L % Tokens != 0:
        raise ValueError(f"Sequence length {L} is not divisible by Tokens={Tokens}")

    F_steps = L // Tokens

    # (B, F, Tokens, D)
    a = a.view(B, F_steps, Tokens, D)
    b = b.view(B, F_steps, Tokens, D)

    # Pool token dimension -> (B, F, D)
    if pool == "mean":
        a = a.mean(dim=2)
        b = b.mean(dim=2)
    elif pool == "max":
        a = a.max(dim=2).values
        b = b.max(dim=2).values
    else:
        raise ValueError(f"Unknown pool type: {pool}")

    if normalize_before_delta:
        a = F.normalize(a, dim=-1, eps=eps)
        b = F.normalize(b, dim=-1, eps=eps)

    # Temporal delta: (B, F-1, D)
    delta_a = a[:, 1:] - a[:, :-1]
    delta_b = b[:, 1:] - b[:, :-1]

    if delta_a.shape[1] == 0:
        return torch.zeros((), device=a.device, dtype=a.dtype)

    if normalize_after_delta:
        delta_a = F.normalize(delta_a, dim=-1, eps=eps)
        delta_b = F.normalize(delta_b, dim=-1, eps=eps)

    # Cosine similarity over delta features
    cos_sim = (delta_a * delta_b).sum(dim=-1)   # (B, F-1)
    loss = 1.0 - cos_sim.mean()

    return loss

def motion_incremental_alignment_tokenwise(
    a: torch.Tensor,
    b: torch.Tensor,
    Tokens=192,
    motion_weight=None,
    eps=1e-8,
):
    """
    Token-wise motion incremental alignment.

    Args:
        a, b: (B, F*Tokens, D). b is treated as the fixed target.

    Returns:
        loss: scalar
    """
    if a.shape != b.shape:
        raise ValueError(f"a and b must have same shape, got {a.shape} vs {b.shape}")

    B, L, D = a.shape
    if L % Tokens != 0:
        raise ValueError(f"Sequence length {L} is not divisible by Tokens={Tokens}")

    F_steps = L // Tokens

    # (B, F, Tokens, D)
    a = a.view(B, F_steps, Tokens, D)
    b = b.detach().view(B, F_steps, Tokens, D)

    # temporal delta per token
    delta_a = a[:, 1:] - a[:, :-1]   # (B, F-1, Tokens, D)
    delta_b = b[:, 1:] - b[:, :-1]

    if delta_a.shape[1] == 0:
        return torch.zeros((), device=a.device, dtype=a.dtype)

    delta_a = F.normalize(delta_a, dim=-1, eps=eps)
    delta_b = F.normalize(delta_b, dim=-1, eps=eps)

    cos_sim = (delta_a * delta_b).sum(dim=-1)   # (B, F-1, Tokens)
    loss_each_token = (1.0 - cos_sim).mean(dim=1)   # (B, Tokens)

    if motion_weight is None:
        loss = loss_each_token.mean()
    else:
        if motion_weight.shape != loss_each_token.shape:
            raise ValueError(
                "motion_weight must have shape [B, Tokens], "
                f"got {motion_weight.shape} for token loss shape {loss_each_token.shape}"
            )
        motion_weight = motion_weight.detach().to(
            device=loss_each_token.device,
            dtype=loss_each_token.dtype,
        )
        loss = (motion_weight * loss_each_token).sum() / (motion_weight.sum() + eps)

    return loss


def build_topk_dest_prob(
    dest_score,
    topk_ratio=0.1,
    tau=3.0,
    eps=1e-6,
):
    """
    Build a destination-token distribution over the top-scoring tokens.

    Args:
        dest_score: [B, T]
    """
    if dest_score.dim() != 2:
        raise ValueError(f"Expected dest_score with shape [B, T], got {dest_score.shape}")

    _, tokens = dest_score.shape
    k = max(1, min(tokens, int(math.ceil(tokens * topk_ratio))))
    topk_idx = torch.topk(dest_score, k, dim=-1).indices

    masked_score = dest_score.new_full(dest_score.shape, -torch.inf)
    masked_score.scatter_(dim=-1, index=topk_idx, src=dest_score.gather(-1, topk_idx))
    prob = torch.softmax(masked_score / max(tau, eps), dim=-1)
    return prob / (prob.sum(dim=-1, keepdim=True) + eps)


def reshape_token_sequence(x, tokens_per_frame, name):
    if x.dim() == 3:
        B, seq_len, _ = x.shape
        if seq_len % tokens_per_frame != 0:
            raise ValueError(
                f"{name} seq_len={seq_len} is not divisible by tokens_per_frame={tokens_per_frame}"
            )
        return x.reshape(B, seq_len // tokens_per_frame, tokens_per_frame, -1)

    if x.dim() == 4:
        return x

    raise ValueError(
        f"Expected {name} with shape [B,F,T,D] or [B,F*T,D], got {x.shape}"
    )


def compute_moving_score(moving_score_source, tokens_per_frame, eps=1e-6):
    moving_score_source = reshape_token_sequence(
        moving_score_source,
        tokens_per_frame,
        "moving_score_source",
    )

    if moving_score_source.shape[1] < 2:
        return moving_score_source.new_zeros(
            moving_score_source.shape[0],
            moving_score_source.shape[2],
        )

    return (
        moving_score_source[:, 1:].detach().float()
        - moving_score_source[:, :-1].detach().float()
    ).norm(dim=-1).mean(dim=1)

def source_conditioned_destination_loss(
    h: torch.Tensor,
    ta: torch.Tensor,
    tokens_per_frame: int,
    moving_score_source: torch.Tensor | None = None,
    moving_score: torch.Tensor | None = None,
    src_prob: torch.Tensor | None = None,
    tau_src: float = 3.0,
    tau_sim: float = 0.2,
    topk_ratio: float = 0.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    if h.shape != ta.shape:
        raise ValueError(f"h and ta must have same shape, got {h.shape} vs {ta.shape}")
 
    h = reshape_token_sequence(h, tokens_per_frame, "h")
    ta = reshape_token_sequence(ta, tokens_per_frame, "ta")
 
    if h.shape[1] < 2:
        return h.new_zeros(())

    if src_prob is None:
        if moving_score is None:
            moving_score_source = ta if moving_score_source is None else moving_score_source
            moving_score = compute_moving_score(
                moving_score_source,
                tokens_per_frame,
                eps=eps,
            )
        else:
            if moving_score.shape != h.shape[:1] + h.shape[2:3]:
                raise ValueError(
                    "moving_score must have shape [B, Tokens], "
                    f"got {moving_score.shape} for h shape {h.shape}"
                )

        src_prob = build_topk_dest_prob(
            moving_score,
            topk_ratio=topk_ratio,
            tau=tau_src,
            eps=eps,
        )
    else:
        if src_prob.shape != h.shape[:1] + h.shape[2:3]:
            raise ValueError(
                "src_prob must have shape [B, Tokens], "
                f"got {src_prob.shape} for h shape {h.shape}"
            )

    src_prob = src_prob.detach()
 
    # 2. H predicted source-to-destination distribution p_h(j | i).
    # Source: h at first frame.
    # Destination: TA at final frame.
    h_src = F.normalize(h[:, 0].float(), dim=-1, eps=eps)
    ta_dest = F.normalize(ta[:, -1].detach().float(), dim=-1, eps=eps)
 
    sim_h = torch.matmul(h_src, ta_dest.transpose(-1, -2))
    log_pred = F.log_softmax(sim_h / max(tau_sim, eps), dim=-1)
 
    # 3. TA teacher source-to-destination distribution q_ta(j | i).
    # No destination norm prior here.
    ta_src = F.normalize(ta[:, 0].detach().float(), dim=-1, eps=eps)
 
    sim_ta = torch.matmul(ta_src, ta_dest.transpose(-1, -2))
    target = F.softmax(sim_ta / max(tau_sim, eps), dim=-1).detach()
 
    # 4. Source-weighted KL distillation.
    kl_each = F.kl_div(
        log_pred,
        target,
        reduction="none",
    ).sum(dim=-1)
 
    loss = (src_prob * kl_each).sum() / (src_prob.sum() + eps)
 
    return loss * 0.5


def unified_dest_and_motion(
    h,
    ta,
    moving_score_source=None,
    Tokens=192,
    tau_sim=0.1,
    tau_src=3.0,
    topk_ratio=0.3,
    eps=1e-6,
):
    moving_score_source = ta if moving_score_source is None else moving_score_source
    moving_score = compute_moving_score(
        moving_score_source,
        Tokens,
        eps=eps,
    )

    src_prob = build_topk_dest_prob(
        moving_score,
        topk_ratio=topk_ratio,
        tau=tau_src,
        eps=eps,
    ).detach()

    loss_dest = source_conditioned_destination_loss(
        h,
        ta,
        tokens_per_frame=Tokens,
        src_prob=src_prob,
        tau_sim=tau_sim,
        tau_src=tau_src,
        topk_ratio=topk_ratio,
        eps=eps,
    )
    
    loss_motion = motion_incremental_alignment_tokenwise(
        h,
        ta,
        Tokens=Tokens,
        motion_weight=src_prob,
        eps=eps,
    )
    
    return {
        'dest_loss': loss_dest,
        'motion_loss': loss_motion,
    }
