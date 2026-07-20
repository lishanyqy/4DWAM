import json
import os
import warnings


def resolve_attn_mode(is_train: bool = False) -> str:
    """Training uses flex attention; inference prefers flashattn."""
    return "flex" if is_train else "flashattn"


def modelswitch(path, is_train: bool = False):
    """
    Legacy helper that used to rewrite transformer/config.json in-place.

    That is unsafe under multi-process launch (torchrun): all ranks open and
    rewrite the same shared HuggingFace cache file, which can leave an empty
    or half-written JSON and raise JSONDecodeError.

    Prefer passing attn_mode into load_transformer / from_pretrained instead.
    This function is kept for call-site compatibility and only returns the
    desired attn_mode without mutating files.
    """
    attn_mode = resolve_attn_mode(is_train)
    config_path = os.path.join(path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config_data = json.load(config_file)
            current_mode = config_data.get("attn_mode")
            if current_mode is not None and current_mode != attn_mode:
                warnings.warn(
                    f"Not rewriting {config_path} (would set attn_mode="
                    f"{attn_mode!r}, file has {current_mode!r}). Pass "
                    f"attn_mode={attn_mode!r} to from_pretrained instead.",
                    stacklevel=2,
                )
        except json.JSONDecodeError as decode_error:
            warnings.warn(
                f"Invalid JSON at {config_path}: {decode_error}. "
                "Re-download the model cache or restore config.json.",
                stacklevel=2,
            )
    return attn_mode
