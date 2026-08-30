#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block-based Prefill -> KV handoff -> Decode experiment."
    )
    parser.add_argument("--seq", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--pool-blocks", type=int, default=8)
    return parser.parse_args()


def init_dist() -> tuple[int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Launch with: torchrun --standalone --nproc-per-node=2 mini_block_kv_handoff.py"
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("This lab intentionally uses exactly 2 ranks.")

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, device


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
    return torch.softmax(scores, dim=-1) @ v


def main() -> None:
    args = parse_args()
    rank, device = init_dist()
    num_blocks = (args.seq + args.block_size - 1) // args.block_size
    if num_blocks > args.pool_blocks:
        raise ValueError("--pool-blocks is too small for this prompt")

    torch.manual_seed(23)
    prompt = torch.randn(args.seq, args.hidden, device=device)
    decode_token = torch.randn(1, args.hidden, device=device)
    wq = torch.randn(args.hidden, args.hidden, device=device) / args.hidden**0.5
    wk = torch.randn(args.hidden, args.hidden, device=device) / args.hidden**0.5
    wv = torch.randn(args.hidden, args.hidden, device=device) / args.hidden**0.5
    request_id = 101

    if rank == 0:
        # Producer: calculate prompt KV and describe the transfer with metadata.
        k_prompt = prompt @ wk
        v_prompt = prompt @ wv
        metadata = torch.tensor(
            [request_id, args.seq, args.hidden, args.block_size, num_blocks],
            dtype=torch.int64,
            device=device,
        )
        dist.send(metadata, dst=1)

        for logical_block in range(num_blocks):
            start = logical_block * args.block_size
            end = min(args.seq, start + args.block_size)
            valid_tokens = end - start
            header = torch.tensor(
                [logical_block, valid_tokens], dtype=torch.int64, device=device
            )
            dist.send(header, dst=1)

            # Fixed-size wire blocks make the final partial block explicit.
            k_block = torch.zeros(args.block_size, args.hidden, device=device)
            v_block = torch.zeros_like(k_block)
            k_block[:valid_tokens] = k_prompt[start:end]
            v_block[:valid_tokens] = v_prompt[start:end]
            dist.send(k_block, dst=1)
            dist.send(v_block, dst=1)

    else:
        # Consumer: receive control metadata first, then place wire blocks into
        # arbitrary physical slots in a local KV pool.
        metadata = torch.empty(5, dtype=torch.int64, device=device)
        dist.recv(metadata, src=0)
        rid, seq, hidden, block_size, remote_num_blocks = [
            int(x) for x in metadata.tolist()
        ]
        assert (rid, seq, hidden, block_size, remote_num_blocks) == (
            request_id,
            args.seq,
            args.hidden,
            args.block_size,
            num_blocks,
        )

        pool_k = torch.empty(
            args.pool_blocks, args.block_size, args.hidden, device=device
        )
        pool_v = torch.empty_like(pool_k)

        # Deliberately allocate from the far end so logical block ids and
        # physical slots are visibly different.
        free_physical_ids = list(reversed(range(args.pool_blocks)))
        block_table: list[int] = []
        valid_per_block: list[int] = []

        for _ in range(num_blocks):
            header = torch.empty(2, dtype=torch.int64, device=device)
            dist.recv(header, src=0)
            logical_block, valid_tokens = [int(x) for x in header.tolist()]
            if logical_block != len(block_table):
                raise RuntimeError("Unexpected logical block order")

            physical_block = free_physical_ids.pop(0)
            k_block = torch.empty(args.block_size, args.hidden, device=device)
            v_block = torch.empty_like(k_block)
            dist.recv(k_block, src=0)
            dist.recv(v_block, src=0)
            pool_k[physical_block] = k_block
            pool_v[physical_block] = v_block
            block_table.append(physical_block)
            valid_per_block.append(valid_tokens)

        # Attention follows the logical block table, not physical memory order.
        remote_k = torch.cat(
            [pool_k[p, :valid] for p, valid in zip(block_table, valid_per_block)],
            dim=0,
        )
        remote_v = torch.cat(
            [pool_v[p, :valid] for p, valid in zip(block_table, valid_per_block)],
            dim=0,
        )

        q_step = decode_token @ wq
        k_step = decode_token @ wk
        v_step = decode_token @ wv
        handed_out = attention(
            q_step,
            torch.cat([remote_k, k_step], dim=0),
            torch.cat([remote_v, v_step], dim=0),
        )

        # Reference: recompute the exact same prompt KV locally.
        ref_out = attention(
            q_step,
            torch.cat([prompt @ wk, k_step], dim=0),
            torch.cat([prompt @ wv, v_step], dim=0),
        )
        max_error = (handed_out - ref_out).abs().max().item()

        logical_bytes = 2 * args.seq * args.hidden * prompt.element_size()
        padded_wire_bytes = (
            2 * num_blocks * args.block_size * args.hidden * prompt.element_size()
        )
        print("=== Mini Block KV Handoff ===")
        print(f"backend={dist.get_backend()} device={device.type}")
        print(f"request_id           : {rid}")
        print(f"logical blocks       : {list(range(num_blocks))}")
        print(f"physical block table : {block_table}")
        print(f"valid tokens/block   : {valid_per_block}")
        print(f"logical KV bytes     : {logical_bytes}")
        print(f"padded wire bytes    : {padded_wire_bytes}")
        print(f"max |block-ref|      : {max_error:.3e}")
        print("PASS" if max_error < 1e-5 else "FAIL")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
