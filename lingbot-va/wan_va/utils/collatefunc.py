import torch
import torch.nn.functional as F


def _pad_tensor_to_shape(tensor, target_shape):
    """Right-pad a tensor with zeros to `target_shape`."""
    if len(tensor.shape) != len(target_shape):
        raise ValueError(f"Cannot pad tensor with shape {tensor.shape} to {target_shape}")

    pad = []
    for current, target in zip(reversed(tensor.shape), reversed(target_shape)):
        if current > target:
            raise ValueError(f"Current dim {current} is larger than target dim {target}")
        pad.extend([0, target - current])
    return F.pad(tensor, pad)


def _stack_padded(tensors):
    max_shape = [max(t.shape[dim] for t in tensors) for dim in range(tensors[0].dim())]
    return torch.stack([_pad_tensor_to_shape(t, max_shape) for t in tensors])


def collate_get_mask(batch):
    """Collate variable-length LingBot-VA samples.

    Dataset items contain tensors with shapes like:
    - text_emb: (T_text, D)
    - latents: (C, F, H, W)
    - actions/actions_mask: (C, F, N, 1)
    - trace: (F_trace, H, W, C_trace)

    For multi-batch training we right-pad every tensor field to the maximum
    shape in the batch, then stack along a new batch dimension.
    """
    if not batch:
        return {}

    out = {}
    keys = batch[0].keys()
    for key in keys:
        values = [sample[key] for sample in batch]
        if torch.is_tensor(values[0]):
            out[key] = _stack_padded(values)
        else:
            out[key] = values

    out["latents_active_frames"] = torch.tensor(
        [sample["latents"].shape[1] for sample in batch],
        dtype=torch.long,
    )
    if "actions" in batch[0]:
        out["actions_active_frames"] = torch.tensor(
            [sample["actions"].shape[1] for sample in batch],
            dtype=torch.long,
        )
    if "trace" in batch[0]:
        out["trace_active_frames"] = torch.tensor(
            [sample["trace"].shape[0] for sample in batch],
            dtype=torch.long,
        )
    return out
