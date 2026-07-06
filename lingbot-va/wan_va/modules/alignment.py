import math

import torch
import torch.nn.functional as F

def create_future_alignment_mask(F, K, device='cuda'):
    """
    创建未来帧对齐的 mask
    
    Args:
        F: Total F frames
        K: future K frames (a[t] 对齐 b[t+1] 到 b[t+K])
        
    Returns:
        mask: (F, F) bool mask, True represent the loss computation part.
    """
    mask = torch.zeros(F, F, dtype=torch.bool, device=device)
    
    for t in range(F):
        # a[t] align from b[t+1] to b[t+K]
        future_frames = range(t+1, min(t+K+1, F))
        for future_t in future_frames:
            mask[t, future_t] = True
            
    return mask


def future_alignment_loss(a, b, K=3, Tokens=192, temperature=1.0, mask_type='window'):
    """
    Future frame alignment.

    Args:
        a: (B, F*Tokens, D) source representation
        b: (B, F*Tokens, D) target representation
        K: aligned future frames
        Tokens: number of tokens per frame
        temperature: temperature coef
        mask_type: 'triangular' or 'window'

    Returns:
        loss: scalar loss
    """
    B, _, D = a.shape

    a = a.reshape(-1, Tokens, D)
    b = b.reshape(-1, Tokens, D)
    Fa, Fb = a.shape[0], b.shape[0]

    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)

    if B == 1:
        a_norm = a_norm.squeeze(0)
        b_norm = b_norm.squeeze(0)
        cos_sim = torch.einsum('ftd,gtd->ftg', a_norm, b_norm)
    else:
        a_flat = a_norm.view(B, Fa * Tokens, D)
        b_flat = b_norm.view(B, Fb * Tokens, D)
        cos_sim_flat = torch.bmm(a_flat, b_flat.transpose(1, 2))
        cos_sim = cos_sim_flat.view(B, Fa, Tokens, Fb, Tokens)
        cos_sim = cos_sim.diagonal(dim1=2, dim2=4).permute(0, 1, 3, 2)

    if mask_type == 'triangular':
        if Fa != Fb:
            raise ValueError(f"Triangular mask requires Fa == Fb, but got Fa={Fa}, Fb={Fb}")
        frame_mask = torch.tril(torch.ones(Fa, Fb, dtype=torch.bool), diagonal=-1)
    elif mask_type == 'window':
        frame_mask = torch.zeros(Fa, Fb, dtype=torch.bool)
        for i in range(Fa):
            for j in range(i, min(i + K, Fb)):
                frame_mask[i, j] = True
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    frame_mask = frame_mask.to(a.device)

    if B == 1:
        token_mask = frame_mask[:, None, :].expand(Fa, Tokens, Fb)
    else:
        token_mask = frame_mask[None, :, :, None].expand(B, Fa, Fb, Tokens)

    valid_sims = cos_sim[token_mask]

    if len(valid_sims) == 0:
        return torch.tensor(0.0, device=a.device)

    if temperature != 1.0:
        valid_sims = valid_sims / temperature

    return 1 - valid_sims.mean()


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
    a,
    b,
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


def destination_loss_safe(
    h,
    ta,
    Tokens=192,
    tau_dest=3.0,
    tau_sim=0.1,
    tau_src=3.0,
    topk_ratio=0.1,
    eps=1e-6,
):
    """
    Destination alignment loss.

    Args:
        h:  [B, F, T, D] or flattened [B, F*T, D]
        ta: [B, F, T, D] or flattened [B, F*T, D]
    """
    if h.shape != ta.shape:
        raise ValueError(f"h and ta must have same shape, got {h.shape} vs {ta.shape}")

    if h.dim() == 3:
        B, L, D = h.shape
        if L % Tokens != 0:
            raise ValueError(f"Sequence length {L} is not divisible by Tokens={Tokens}")
        h = h.reshape(B, L // Tokens, Tokens, D)
        ta = ta.reshape(B, L // Tokens, Tokens, D)
    elif h.dim() != 4:
        raise ValueError(f"Expected h and ta with shape [B,F,T,D] or [B,F*T,D], got {h.shape}")

    if h.shape[1] == 0:
        return h.new_zeros(())

    h_final = h[:, -1]
    ta_final = ta[:, -1].detach()

    h_final = F.normalize(h_final.float(), dim=-1, eps=eps)
    ta_final = F.normalize(ta_final.float(), dim=-1, eps=eps)

    sim = torch.matmul(h_final, ta_final.transpose(-1, -2))
    log_pred = F.log_softmax(sim / max(tau_sim, eps), dim=-1)

    dest_score = ta[:, -1].detach().float().norm(dim=-1)
    dest_prob = build_topk_dest_prob(
        dest_score,
        topk_ratio=topk_ratio,
        tau=tau_dest,
        eps=eps,
    ).detach()

    target = dest_prob.unsqueeze(1).expand_as(log_pred)

    src_score = (ta[:, -1].detach().float() - ta[:, 0].detach().float()).norm(dim=-1)
    src_weight = F.softmax(src_score / max(tau_src, eps), dim=-1).detach()

    kl_each = F.kl_div(
        log_pred,
        target,
        reduction="none",
    ).sum(dim=-1)

    loss = (src_weight * kl_each).sum() / (src_weight.sum() + eps)
    return loss


# def source_conditioned_destination_loss(
#     h: torch.Tensor,
#     ta: torch.Tensor,
#     tokens_per_frame: int,
#     tau_src: float = 3.0,
#     tau_dest: float = 3.0,
#     tau_sim: float = 0.2,
#     topk_ratio: float = 0.2,
#     eps: float = 1e-6,
# ) -> torch.Tensor:
#     if h.shape != ta.shape:
#         raise ValueError(f"h and ta must have same shape, got {h.shape} vs {ta.shape}")

#     if h.dim() == 3:
#         B, seq_len, D = h.shape
#         if seq_len % tokens_per_frame != 0:
#             raise ValueError(
#                 f"seq_len={seq_len} is not divisible by tokens_per_frame={tokens_per_frame}"
#             )
#         F_ = seq_len // tokens_per_frame
#         h = h.reshape(B, F_, tokens_per_frame, D)
#         ta = ta.reshape(B, F_, tokens_per_frame, D)

#     elif h.dim() == 4:
#         B, F_, T, D = h.shape
#     else:
#         raise ValueError(f"Expected [B,F,T,D] or [B,F*T,D], got {h.shape}")

#     if h.shape[1] < 2:
#         return h.new_zeros(())

#     # 1. source motion weights: which start tokens are worth supervising
#     moving_score = (ta[:, 1:].detach().float() - ta[:, :-1].detach().float()).norm(dim=-1).mean(dim=1)

#     src_prob = build_topk_dest_prob(
#         moving_score,
#         topk_ratio=topk_ratio,
#         tau=tau_src,
#         eps=eps,
#     ).detach()

#     # 2. destination saliency prior: which final tokens are likely endpoints / important regions
#     dest_score = ta[:, -1].detach().float().norm(dim=-1)

#     dest_prob = build_topk_dest_prob(
#         dest_score,
#         topk_ratio=topk_ratio,
#         tau=tau_dest,
#         eps=eps,
#         largest = False,
#     ).detach()

#     # 3. h predicted source-to-destination distribution p(j|i)
#     h_src = F.normalize(h[:, 0].float(), dim=-1, eps=eps)
#     ta_dest = F.normalize(ta[:, -1].detach().float(), dim=-1, eps=eps)

#     sim_h = torch.matmul(h_src, ta_dest.transpose(-1, -2))
#     log_pred = F.log_softmax(sim_h / max(tau_sim, eps), dim=-1)

#     # 4. TA source-conditioned target distribution q(j|i)
#     ta_src = F.normalize(ta[:, 0].detach().float(), dim=-1, eps=eps)

#     sim_ta = torch.matmul(ta_src, ta_dest.transpose(-1, -2))

#     target_logits = (
#         sim_ta / max(tau_sim, eps)
#         + torch.log(dest_prob.unsqueeze(1) + eps)
#     )

#     target = F.softmax(target_logits, dim=-1).detach()

#     # 5. distillation loss, weighted by moving source tokens
#     kl_each = F.kl_div(
#         log_pred,
#         target,
#         reduction="none",
#     ).sum(dim=-1)

#     loss = (src_prob * kl_each).sum() / (src_prob.sum() + eps)
#     return loss * 0.5

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
