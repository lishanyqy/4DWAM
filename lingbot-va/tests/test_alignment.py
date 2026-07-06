from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch


def _load_alignment_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "wan_va" / "modules" / "alignment.py"
    )
    spec = spec_from_file_location("alignment_under_test", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


alignment = _load_alignment_module()


def test_motion_incremental_alignment_returns_zero_for_identical_inputs():
    batch_size = 2
    frames = 4
    tokens = 3
    hidden_dim = 5

    x = torch.randn(batch_size, frames * tokens, hidden_dim)
    loss = alignment.motion_incremental_alignment(x, x.clone(), Tokens=tokens)

    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, torch.tensor(0.0, dtype=loss.dtype), atol=1e-6)


@pytest.mark.parametrize("pool", ["mean", "max"])
def test_motion_incremental_alignment_runs_with_valid_inputs(pool):
    # Construct frame-wise motion so temporal deltas are non-zero and stable.
    frame_features = torch.tensor(
        [[[0.0, 1.0], [1.0, 3.0], [3.0, 6.0], [6.0, 10.0]]], dtype=torch.float32
    )
    a = frame_features.repeat_interleave(2, dim=1)
    b = a.clone()

    loss = alignment.motion_incremental_alignment(a, b, Tokens=2, pool=pool)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() <= 1e-6


def test_motion_incremental_alignment_returns_zero_when_only_one_frame():
    x = torch.randn(2, 4, 6)

    loss = alignment.motion_incremental_alignment(x, x.clone(), Tokens=4)

    assert loss.shape == torch.Size([])
    assert loss.item() == 0.0


def test_motion_incremental_alignment_raises_for_invalid_sequence_length():
    x = torch.randn(1, 5, 4)

    with pytest.raises(ValueError, match="not divisible by Tokens"):
        alignment.motion_incremental_alignment(x, x.clone(), Tokens=2)


def test_motion_incremental_alignment_raises_for_unknown_pool():
    x = torch.randn(1, 6, 4)

    with pytest.raises(ValueError, match="Unknown pool type"):
        alignment.motion_incremental_alignment(x, x.clone(), Tokens=3, pool="sum")


def test_motion_incremental_alignment_tokenwise_uses_motion_weight():
    a = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
        dtype=torch.float32,
    )
    b = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]],
        dtype=torch.float32,
    )

    static_token_weight = torch.tensor([[1.0, 0.0]])
    moving_token_weight = torch.tensor([[0.0, 1.0]])

    static_loss = alignment.motion_incremental_alignment_tokenwise(
        a,
        b,
        Tokens=2,
        motion_weight=static_token_weight,
    )
    moving_loss = alignment.motion_incremental_alignment_tokenwise(
        a,
        b,
        Tokens=2,
        motion_weight=moving_token_weight,
    )

    assert torch.isclose(static_loss, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(moving_loss, torch.tensor(2.0), atol=1e-6)


def test_build_topk_dest_prob_keeps_only_top_tokens():
    score = torch.tensor([[1.0, 4.0, 2.0, 3.0]])

    prob = alignment.build_topk_dest_prob(score, topk_ratio=0.5, tau=1.0)

    assert prob.shape == score.shape
    assert torch.isclose(prob.sum(), torch.tensor(1.0), atol=1e-6)
    assert prob[0, 0].item() == 0.0
    assert prob[0, 2].item() == 0.0
    assert prob[0, 1].item() > 0.0
    assert prob[0, 3].item() > 0.0


def test_destination_loss_safe_runs_and_backpropagates():
    h = torch.randn(2, 3, 4, 5, requires_grad=True)
    ta = torch.randn(2, 3, 4, 5)

    loss = alignment.destination_loss_safe(h, ta, Tokens=4)

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    loss.backward()
    assert h.grad is not None


def test_destination_loss_safe_accepts_flat_sequence():
    h = torch.randn(1, 6, 3, requires_grad=True)
    ta = torch.randn(1, 6, 3)

    loss = alignment.destination_loss_safe(h, ta, Tokens=2)

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


def test_unified_dest_and_motion_accepts_pre_projection_moving_score_source():
    h = torch.randn(1, 6, 5, requires_grad=True)
    raw_trace = torch.randn(1, 6, 7, requires_grad=True)
    trace_proj = torch.nn.Linear(7, 5)
    ta = trace_proj(raw_trace)

    original_build_topk_dest_prob = alignment.build_topk_dest_prob
    build_topk_call_count = 0

    def wrapped_build_topk_dest_prob(*args, **kwargs):
        nonlocal build_topk_call_count
        build_topk_call_count += 1
        return original_build_topk_dest_prob(*args, **kwargs)

    alignment.build_topk_dest_prob = wrapped_build_topk_dest_prob

    try:
        losses = alignment.unified_dest_and_motion(
            h,
            ta,
            moving_score_source=raw_trace,
            Tokens=2,
        )
    finally:
        alignment.build_topk_dest_prob = original_build_topk_dest_prob

    assert set(losses) == {"dest_loss", "motion_loss"}
    assert torch.isfinite(losses["dest_loss"])
    assert torch.isfinite(losses["motion_loss"])
    assert build_topk_call_count == 1

    total_loss = losses["dest_loss"] + losses["motion_loss"]
    total_loss.backward()

    assert h.grad is not None
    assert raw_trace.grad is None
    assert all(param.grad is None for param in trace_proj.parameters())
