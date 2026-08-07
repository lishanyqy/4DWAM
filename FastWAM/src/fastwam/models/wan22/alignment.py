import math

import torch
import torch.nn.functional as F


def build_topk_dest_prob(
    dest_score: torch.Tensor,
    topk_ratio: float = 0.1,
    tau: float = 3.0,
    eps: float = 1e-6,
    largest = True,
) -> torch.Tensor:
    if dest_score.dim() != 2:
        raise ValueError(f"Expected dest_score with shape [B, T], got {tuple(dest_score.shape)}")

    _, tokens = dest_score.shape
    k = max(1, min(tokens, int(math.ceil(tokens * topk_ratio))))
    topk_idx = torch.topk(dest_score, k, dim=-1, largest = largest).indices

    masked_score = dest_score.new_full(dest_score.shape, -torch.inf)
    masked_score.scatter_(dim=-1, index=topk_idx, src=dest_score.gather(-1, topk_idx))
    prob = torch.softmax(masked_score / max(tau, eps), dim=-1)
    return prob / (prob.sum(dim=-1, keepdim=True) + eps)


def source_conditioned_destination_loss(
    h: torch.Tensor,
    ta: torch.Tensor,
    tokens_per_frame: int,
    tau_src: float = 3.0,
    tau_dest: float = 3.0,
    tau_sim: float = 0.2,
    topk_ratio: float = 0.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    if h.shape != ta.shape:
        raise ValueError(f"h and ta must have same shape, got {h.shape} vs {ta.shape}")

    if h.dim() == 3:
        B, seq_len, D = h.shape
        if seq_len % tokens_per_frame != 0:
            raise ValueError(
                f"seq_len={seq_len} is not divisible by tokens_per_frame={tokens_per_frame}"
            )
        F_ = seq_len // tokens_per_frame
        h = h.reshape(B, F_, tokens_per_frame, D)
        ta = ta.reshape(B, F_, tokens_per_frame, D)

    elif h.dim() == 4:
        B, F_, T, D = h.shape
    else:
        raise ValueError(f"Expected [B,F,T,D] or [B,F*T,D], got {h.shape}")

    if h.shape[1] < 2:
        return h.new_zeros(())

    # 1. source motion weights: which start tokens are worth supervising
    moving_score = (ta[:, 1:].detach().float() - ta[:, :-1].detach().float()).norm(dim=-1).mean(dim=1)

    src_prob = build_topk_dest_prob(
        moving_score,
        topk_ratio=topk_ratio,
        tau=tau_src,
        eps=eps,
    ).detach()

    # 2. destination saliency prior: which final tokens are likely endpoints / important regions
    dest_score = ta[:, -1].detach().float().norm(dim=-1)

    dest_prob = build_topk_dest_prob(
        dest_score,
        topk_ratio=topk_ratio,
        tau=tau_dest,
        eps=eps,
        largest = False,
    ).detach()

    # 3. h predicted source-to-destination distribution p(j|i)
    h_src = F.normalize(h[:, 0].float(), dim=-1, eps=eps)
    ta_dest = F.normalize(ta[:, -1].detach().float(), dim=-1, eps=eps)

    sim_h = torch.matmul(h_src, ta_dest.transpose(-1, -2))
    log_pred = F.log_softmax(sim_h / max(tau_sim, eps), dim=-1)

    # 4. TA source-conditioned target distribution q(j|i)
    ta_src = F.normalize(ta[:, 0].detach().float(), dim=-1, eps=eps)

    sim_ta = torch.matmul(ta_src, ta_dest.transpose(-1, -2))

    target_logits = (
        sim_ta / max(tau_sim, eps)
        + torch.log(dest_prob.unsqueeze(1) + eps)
    )

    target = F.softmax(target_logits, dim=-1).detach()

    # 5. distillation loss, weighted by moving source tokens
    kl_each = F.kl_div(
        log_pred,
        target,
        reduction="none",
    ).sum(dim=-1)

    loss = (src_prob * kl_each).sum() / (src_prob.sum() + eps)
    return loss * 0.5

def destination_loss_safe(
    h: torch.Tensor,
    ta: torch.Tensor,
    tokens_per_frame: int,
    tau_dest: float = 3.0,
    tau_sim: float = 0.1,
    tau_src: float = 3.0,
    topk_ratio: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    if h.shape != ta.shape:
        raise ValueError(f"h and ta must have same shape, got {tuple(h.shape)} vs {tuple(ta.shape)}")

    if h.dim() == 3:
        batch_size, seq_len, dim = h.shape
        if seq_len % tokens_per_frame != 0:
            raise ValueError(f"Sequence length {seq_len} is not divisible by tokens_per_frame={tokens_per_frame}")
        h = h.reshape(batch_size, seq_len // tokens_per_frame, tokens_per_frame, dim)
        ta = ta.reshape(batch_size, seq_len // tokens_per_frame, tokens_per_frame, dim)
    elif h.dim() != 4:
        raise ValueError(f"Expected h and ta with shape [B,F,T,D] or [B,F*T,D], got {tuple(h.shape)}")

    if h.shape[1] == 0:
        return h.new_zeros(())

    h_final = F.normalize(h[:, -1].float(), dim=-1, eps=eps)
    ta_final = F.normalize(ta[:, -1].detach().float(), dim=-1, eps=eps)

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

    kl_each = F.kl_div(log_pred, target, reduction="none").sum(dim=-1)
    return (src_weight * kl_each).sum() / (src_weight.sum() + eps)


def motion_incremental_alignment_tokenwise(
    h: torch.Tensor,
    ta: torch.Tensor,
    tokens_per_frame: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    if h.shape != ta.shape:
        raise ValueError(f"h and ta must have same shape, got {tuple(h.shape)} vs {tuple(ta.shape)}")
    if h.dim() != 3:
        raise ValueError(f"Expected h and ta to be 3D [B,F*T,D], got {tuple(h.shape)}")

    batch_size, seq_len, dim = h.shape
    if seq_len % tokens_per_frame != 0:
        raise ValueError(f"Sequence length {seq_len} is not divisible by tokens_per_frame={tokens_per_frame}")

    num_frames = seq_len // tokens_per_frame
    h = h.view(batch_size, num_frames, tokens_per_frame, dim)
    ta = ta.detach().view(batch_size, num_frames, tokens_per_frame, dim)

    delta_h = h[:, 1:] - h[:, :-1]
    delta_ta = ta[:, 1:] - ta[:, :-1]
    if delta_h.shape[1] == 0:
        return h.new_zeros(())

    delta_h = F.normalize(delta_h, dim=-1, eps=eps)
    delta_ta = F.normalize(delta_ta, dim=-1, eps=eps)
    cos_sim = (delta_h * delta_ta).sum(dim=-1)
    return 1.0 - cos_sim.mean()


def unified_dest_and_motion(
    h: torch.Tensor,
    ta: torch.Tensor,
    tokens_per_frame: int,
    tau_dest: float = 3.0,
    tau_sim: float = 0.1,
    tau_src: float = 3.0,
    topk_ratio: float = 0.2,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    # loss_dest = destination_loss_safe(
    #     h,
    #     ta,
    #     tokens_per_frame=tokens_per_frame,
    #     tau_dest=tau_dest,
    #     tau_sim=tau_sim,
    #     tau_src=tau_src,
    #     topk_ratio=topk_ratio,
    #     eps=eps,
    # )
    loss_dest = source_conditioned_destination_loss(
        h,
        ta,
        tokens_per_frame=tokens_per_frame,
        tau_dest=tau_dest,
        tau_sim=tau_sim,
        tau_src=tau_src,
        topk_ratio=topk_ratio,
        eps=eps,
    )
    
    loss_motion = motion_incremental_alignment_tokenwise(
        h,
        ta,
        tokens_per_frame=tokens_per_frame,
        eps=eps,
    )
    return {
        "dest_loss": loss_dest,
        "motion_loss": loss_motion,
    }
