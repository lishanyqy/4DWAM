#!/usr/bin/env python3
"""Generate empty text embedding for CFG dropout.

Matches latent preprocessing in extract_latents_from_pixels_adaptv1.encode_text:
encode the empty string with UMT5, keep active tokens, then zero-pad to
``text_length`` (default 128). This is not a slice of a longer empty_emb.pt.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import T5TokenizerFast, UMT5EncoderModel


@torch.no_grad()
def encode_empty_text(
    text_encoder: UMT5EncoderModel,
    tokenizer: T5TokenizerFast,
    device: torch.device,
    text_length: int,
    prompt: str = "",
) -> torch.Tensor:
    """Encode ``prompt`` (default empty) and zero-pad to ``text_length``.

    Returns:
        text_emb: float/bfloat16 tensor of shape [text_length, hidden_dim]
    """
    if text_length <= 0:
        raise ValueError(f"text_length must be positive, got {text_length}")

    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=text_length,
        add_special_tokens=True,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}

    encoder_output = text_encoder(**tokens)
    last_hidden_state = encoder_output.last_hidden_state[0]  # [L, D]
    attention_mask = tokens["attention_mask"][0]
    active_token_count = int(attention_mask.sum().item())
    if active_token_count <= 0:
        raise RuntimeError("Tokenizer produced zero active tokens for empty prompt.")
    if active_token_count > text_length:
        raise RuntimeError(
            f"Active token count {active_token_count} exceeds text_length={text_length}."
        )

    active_text_embedding = last_hidden_state[:active_token_count].to(torch.bfloat16)
    if active_text_embedding.shape[0] == text_length:
        return active_text_embedding.detach().cpu().contiguous()

    zero_padding = torch.zeros(
        text_length - active_text_embedding.shape[0],
        active_text_embedding.shape[1],
        dtype=active_text_embedding.dtype,
        device=active_text_embedding.device,
    )
    padded_text_embedding = torch.cat([active_text_embedding, zero_padding], dim=0)
    return padded_text_embedding.detach().cpu().contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed-length empty text embedding for CFG dropout."
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default=os.path.join(
            os.environ.get("CACHE_PATH", "/soft/wangxi/.cache"),
            "huggingface/hub/models--robbyant--lingbot-va-base/"
            "snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c",
        ),
        help="Directory containing text_encoder/ and tokenizer/.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Where to write empty_emb.pt",
    )
    parser.add_argument(
        "--text-length",
        type=int,
        default=128,
        help="Padded sequence length. Must match latent text_emb length.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt to encode as the unconditional embedding (default empty string).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_root = Path(args.models_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    device = torch.device(args.device)

    text_encoder_path = models_root / "text_encoder"
    tokenizer_path = models_root / "tokenizer"
    if not text_encoder_path.is_dir():
        raise FileNotFoundError(f"Missing text_encoder directory: {text_encoder_path}")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Missing tokenizer directory: {tokenizer_path}")

    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = T5TokenizerFast.from_pretrained(str(tokenizer_path))
    print(f"Loading text encoder from {text_encoder_path}")
    text_encoder = UMT5EncoderModel.from_pretrained(
        str(text_encoder_path),
        torch_dtype=torch.bfloat16,
    ).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    empty_embedding = encode_empty_text(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=device,
        text_length=args.text_length,
        prompt=args.prompt,
    )

    if empty_embedding.ndim != 2:
        raise RuntimeError(f"Expected 2D empty embedding, got {tuple(empty_embedding.shape)}")
    if empty_embedding.shape[0] != args.text_length:
        raise RuntimeError(
            f"Expected length {args.text_length}, got {empty_embedding.shape[0]}"
        )
    if empty_embedding.shape[1] != 4096:
        raise RuntimeError(
            f"Expected hidden dim 4096, got {empty_embedding.shape[1]}"
        )

    active_row_count = int((empty_embedding.abs().sum(dim=1) > 0).sum().item())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(empty_embedding, output_path)

    print(f"Saved empty_emb to {output_path}")
    print(f"  shape={tuple(empty_embedding.shape)} dtype={empty_embedding.dtype}")
    print(f"  active_nonzero_rows={active_row_count}")
    print(f"  prompt={args.prompt!r}")


if __name__ == "__main__":
    main()
