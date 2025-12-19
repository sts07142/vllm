#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generate a cascade-attention heuristic decision matrix and dump to CSV.

Usage examples:
  # Default sweep (FA2+A100-ish, FA3+H100-ish)
  python tools/fa_heruistic/flash_attn_cascade_dump.py --out /tmp/cascade_matrix_all.csv

  # Only FA2 with custom num_sms
  python tools/fa_heruistic/flash_attn_cascade_dump.py --fa-version 2 --num-sms 128 --out /tmp/fa2_4090.csv

  # Only FA3 with custom num_sms
  python tools/fa_heruistic/flash_attn_cascade_dump.py --fa-version 3 --num-sms 120 --out /tmp/fa3_h100.csv

The matrix is derived from the same sweep logic used in pytest, so results can
be plotted or compared across versions/branches.
"""  # noqa: E501

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from vllm.v1.attention.backends import flash_attn as fa_mod


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


# ---------------------------------------------------------------------------
# Local copy of sweep logic to avoid importing from tests.
# Also emit approximate cascade_time / flash_time for analysis.
# ---------------------------------------------------------------------------
def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
    dcp_world_size: int,
    head_size: int | None = None,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    fa_version = fa_mod.get_flash_attn_version()
    # Fallback if FA version is unknown; keep previous behavior.
    if fa_version is None:
        fa_version = 2
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    # disable cascade attention for DCP
    if dcp_world_size > 1:
        return False

    def _fa2_use_cascade() -> bool:
        # Heuristics to decide whether using cascade attention is beneficial.
        # 1. When FlashDecoding is not used for normal attention, cascade
        #    attention is likely to be faster since it saves memory bandwidth.
        num_queries_per_kv = num_query_heads // num_kv_heads
        # The criteria for using FlashDecoding can be found in the following
        # link:
        # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
        use_flash_decoding = (
            num_queries_per_kv > 1
            and not use_sliding_window
            and not use_alibi
            and np.all(query_lens == 1)
        )
        if not use_flash_decoding:
            # Use cascade attention.
            return True

        # 2. When FlashDecoding is used for normal attention, it is not clear
        #    whether cascade attention is beneficial, because FlashDecoding can
        #    launch more CTAs than cascade attention.
        #    We use a simple performance model to compare the two methods.
        #    NOTE(woosuk): The performance model is very rough and may not be
        #    accurate.
        num_tokens = num_reqs
        # NOTE(woosuk): These are default tile sizes. flash-attn might use
        # different tile sizes (e.g., 64 or 256) depending on the configuration.
        q_tile_size = 128
        kv_tile_size = 128
        num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

        cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
        cascade_waves = cdiv(cascade_ctas, num_sms)
        cascade_time = cascade_waves * num_prefix_tiles

        flash_decoding_ctas = (
            num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
        )
        flash_decoding_ctas *= num_prefix_tiles
        flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

        # Use cascade attention if it is faster than FlashDecoding.
        return cascade_time < flash_decoding_time

    def _fa3_tile_sizes(h: int) -> tuple[int, int]:
        # Approximate FA3 tile sizes based on hopper tile_size_fwd_sm90
        # (flash-attention/hopper/tile_size.h). Keeps coarse buckets only.
        if h <= 64:
            return 192, 160
        if h <= 96:
            return 192, 144
        if h <= 128:
            return 128, 176
        if h <= 192:
            return 128, 112
        return 128, 80

    def _fa3_use_cascade() -> bool:
        # FA3: incorporate pack_gqa and split heuristics (approximate).
        h = head_size if head_size is not None else 128
        q_tile_size, kv_tile_size = _fa3_tile_sizes(h)
        num_queries_per_kv = num_query_heads // num_kv_heads

        # PackGQA approximation (flash-attention/hopper/flash_api.cpp:get_pack_gqa):
        # when q/kv ratio is large, packing reduces effective kv heads.
        effective_kv_heads = num_kv_heads
        if num_queries_per_kv >= 4:
            effective_kv_heads = max(1, num_kv_heads // 2)

        # Split heuristic approximation (get_num_splits / num_splits_heuristic):
        # if CTAs << SMs, keep split=1; otherwise allow small split factor.
        base_ctas = num_query_heads * cdiv(num_reqs, q_tile_size)
        split_factor = 1
        if base_ctas < int(0.8 * num_sms):
            split_factor = 1
        elif base_ctas < 2 * num_sms:
            split_factor = 2
        else:
            split_factor = 3

        num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

        cascade_ctas = num_query_heads * cdiv(num_reqs, q_tile_size)
        cascade_waves = cdiv(cascade_ctas, num_sms)
        cascade_time = cascade_waves * num_prefix_tiles

        flash_decoding_ctas = (
            num_reqs * effective_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
        )
        flash_decoding_ctas *= num_prefix_tiles
        flash_decoding_time = cdiv(flash_decoding_ctas, num_sms * split_factor)

        return cascade_time < flash_decoding_time

    return _fa2_use_cascade() if fa_version == 2 else _fa3_use_cascade()


def _fa2_times(
    prefix: int,
    batch: int,
    num_query_heads: int,
    num_kv_heads: int,
    num_sms: int,
) -> tuple[int, int]:
    num_queries_per_kv = num_query_heads // num_kv_heads
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(prefix, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(batch, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = batch * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    flash_decoding_time = cdiv(
        flash_decoding_ctas * num_prefix_tiles,
        num_sms,
    )
    return cascade_time, flash_decoding_time


def _fa3_tile_sizes(head_size: int) -> tuple[int, int]:
    # Approximate Hopper tile sizes (see tile_size_fwd_sm90).
    if head_size <= 64:
        return 192, 160
    if head_size <= 96:
        return 192, 144
    if head_size <= 128:
        return 128, 176
    if head_size <= 192:
        return 128, 112
    return 128, 80


def _fa3_times(
    prefix: int,
    batch: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    num_sms: int,
) -> tuple[int, int]:
    q_tile_size, kv_tile_size = _fa3_tile_sizes(head_size)
    num_queries_per_kv = num_query_heads // num_kv_heads

    # PackGQA approximation
    effective_kv_heads = num_kv_heads
    if num_queries_per_kv >= 4:
        effective_kv_heads = max(1, num_kv_heads // 2)

    base_ctas = num_query_heads * cdiv(batch, q_tile_size)
    if base_ctas < int(0.8 * num_sms):
        split_factor = 1
    elif base_ctas < 2 * num_sms:
        split_factor = 2
    else:
        split_factor = 3

    num_prefix_tiles = cdiv(prefix, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(batch, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        batch * effective_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_time = cdiv(
        flash_decoding_ctas * num_prefix_tiles,
        num_sms * split_factor,
    )
    return cascade_time, flash_decoding_time


def _sweep(
    fa_version: int,
    num_sms: int,
    head_sizes: list[int],
    gqa_ratios: list[int],
    prefixes: list[int],
    batches: list[int],
) -> list[dict]:
    rows: list[dict] = []
    num_query_heads = 8
    for head_size in head_sizes:
        for gqa_ratio in gqa_ratios:
            num_kv_heads = max(1, num_query_heads // gqa_ratio)
            for prefix in prefixes:
                for batch in batches:
                    query_lens = np.ones(batch, dtype=np.int32)
                    # defer import to keep dependency local
                    use_cascade = use_cascade_attention(
                        common_prefix_len=prefix,
                        query_lens=query_lens,
                        num_query_heads=num_query_heads,
                        num_kv_heads=num_kv_heads,
                        use_alibi=False,
                        use_sliding_window=False,
                        use_local_attention=False,
                        num_sms=num_sms,
                        dcp_world_size=1,
                        head_size=head_size,
                    )
                    if fa_version == 2:
                        cascade_time, flash_time = _fa2_times(
                            prefix=prefix,
                            batch=batch,
                            num_query_heads=num_query_heads,
                            num_kv_heads=num_kv_heads,
                            num_sms=num_sms,
                        )
                    else:
                        cascade_time, flash_time = _fa3_times(
                            prefix=prefix,
                            batch=batch,
                            num_query_heads=num_query_heads,
                            num_kv_heads=num_kv_heads,
                            head_size=head_size,
                            num_sms=num_sms,
                        )
                    rows.append(
                        {
                            "fa_version": fa_version,
                            "head_size": head_size,
                            "gqa_ratio": gqa_ratio,
                            "prefix": prefix,
                            "batch": batch,
                            "num_sms": num_sms,
                            "num_query_heads": num_query_heads,
                            "num_kv_heads": num_kv_heads,
                            "use_cascade": bool(use_cascade),
                            "cascade_time": cascade_time,
                            "flash_time": flash_time,
                        }
                    )
    return rows


def sweep_once(
    fa_version: int,
    num_sms: int | None = None,
    head_sizes: list[int] | None = None,
    gqa_ratios: list[int] | None = None,
    prefixes: list[int] | None = None,
    batches: list[int] | None = None,
) -> list[dict]:
    if fa_version == 2:
        num_sms = num_sms or 108
        head_sizes = head_sizes or [64, 128]
        gqa_ratios = gqa_ratios or [1, 2, 4]
        prefixes = prefixes or [128, 512, 2048]
        batches = batches or [4, 16, 32]
    elif fa_version == 3:
        num_sms = num_sms or 120
        head_sizes = head_sizes or [64, 96, 128, 192, 256]
        gqa_ratios = gqa_ratios or [1, 2, 4, 8]
        prefixes = prefixes or [128, 256, 512, 2048, 8192]
        batches = batches or [4, 16, 32, 64]
    else:
        raise ValueError(f"Unsupported fa_version: {fa_version}")

    return _sweep(
        fa_version=fa_version,
        num_sms=num_sms,
        head_sizes=head_sizes,
        gqa_ratios=gqa_ratios,
        prefixes=prefixes,
        batches=batches,
    )


def generate_matrix(
    num_sms_fa2: int | None = None,
    num_sms_fa3: int | None = None,
    head_sizes: list[int] | None = None,
    gqa_ratios: list[int] | None = None,
    prefixes: list[int] | None = None,
    batches: list[int] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    rows.extend(
        sweep_once(
            fa_version=2,
            num_sms=num_sms_fa2,
            head_sizes=head_sizes,
            gqa_ratios=gqa_ratios,
            prefixes=prefixes,
            batches=batches,
        )
    )
    rows.extend(
        sweep_once(
            fa_version=3,
            num_sms=num_sms_fa3,
            head_sizes=head_sizes,
            gqa_ratios=gqa_ratios,
            prefixes=prefixes,
            batches=batches,
        )
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fa-version",
        choices=["all", "2", "3"],
        default="all",
        help="Which FA version to sweep (default: all).",
    )
    parser.add_argument(
        "--num-sms",
        type=int,
        default=None,
        help="Override SM count for the sweep",
    )
    parser.add_argument(
        "--head-sizes",
        type=str,
        default=None,
        help="Comma-separated head sizes override (e.g., 64,128,192).",
    )
    parser.add_argument(
        "--gqa-ratios",
        type=str,
        default=None,
        help="Comma-separated GQA ratios override (e.g., 1,2,4,8).",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        default=None,
        help="Comma-separated prefixes override (e.g., 128,512,2048).",
    )
    parser.add_argument(
        "--batches",
        type=str,
        default=None,
        help="Comma-separated batches override (e.g., 4,16,32,64).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    def _parse_list(s: str | None, cast=int):
        if s is None:
            return None
        return [cast(x) for x in s.split(",") if x]

    head_sizes = _parse_list(args.head_sizes)
    gqa_ratios = _parse_list(args.gqa_ratios)
    prefixes = _parse_list(args.prefixes)
    batches = _parse_list(args.batches)

    if args.fa_version == "all":
        rows = generate_matrix(
            num_sms_fa2=args.num_sms,
            num_sms_fa3=args.num_sms,
            head_sizes=head_sizes,
            gqa_ratios=gqa_ratios,
            prefixes=prefixes,
            batches=batches,
        )
    else:
        rows = sweep_once(
            fa_version=int(args.fa_version),
            num_sms=args.num_sms,
            head_sizes=head_sizes,
            gqa_ratios=gqa_ratios,
            prefixes=prefixes,
            batches=batches,
        )

    if not rows:
        raise SystemExit("No rows generated.")

    fieldnames = list(rows[0].keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
