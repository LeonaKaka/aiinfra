#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mini scheduler + local/cold/remote KV lifecycle capstone."
    )
    p.add_argument("--head-dim", type=int, default=6)
    p.add_argument("--block-size", type=int, default=2)
    p.add_argument("--pool-blocks", type=int, default=8)
    p.add_argument("--token-budget", type=int, default=6)
    return p.parse_args()


def init_dist():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch with torchrun --standalone --nproc-per-node=2 ...")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("This capstone expects exactly 2 ranks: producer + consumer")
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, device


def make_kv(prompt: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor):
    return prompt @ wk, prompt @ wv


def decode_one(x_new: torch.Tensor, k: torch.Tensor, v: torch.Tensor, wq: torch.Tensor):
    q = x_new @ wq
    scores = (q @ k.transpose(0, 1)) / math.sqrt(q.shape[-1])
    return torch.softmax(scores, dim=-1) @ v


def deterministic_model_and_requests(device: torch.device, d: int):
    torch.manual_seed(3001)
    wk = torch.randn(d, d, device=device) / d**0.5
    wv = torch.randn(d, d, device=device) / d**0.5
    wq = torch.randn(d, d, device=device) / d**0.5
    remote_prompt = torch.randn(4, d, device=device)
    remote_new = torch.randn(1, d, device=device)
    local_prompt = torch.randn(2, d, device=device)
    local_new = torch.randn(1, d, device=device)
    cold_prompt = torch.randn(4, d, device=device)
    return wk, wv, wq, remote_prompt, remote_new, local_prompt, local_new, cold_prompt


def producer(args: argparse.Namespace, device: torch.device) -> None:
    wk, wv, _, remote_prompt, _, _, _, _ = deterministic_model_and_requests(
        device, args.head_dim
    )
    k, v = make_kv(remote_prompt, wk, wv)
    if remote_prompt.shape[0] % args.block_size != 0:
        raise RuntimeError("remote prompt must fill whole teaching blocks")
    num_blocks = remote_prompt.shape[0] // args.block_size
    blocks = torch.stack([k, v], dim=1).view(
        num_blocks, args.block_size, 2, args.head_dim
    ).contiguous()
    payload = blocks.flatten().contiguous()

    request_id = 202
    generation = 7
    model_signature = 424242
    descriptor = torch.tensor(
        [
            request_id,
            remote_prompt.shape[0],
            args.block_size,
            num_blocks,
            generation,
            model_signature,
            args.head_dim,
            payload.numel(),
        ],
        dtype=torch.int64,
        device=device,
    )

    # Control plane first: the consumer must know whether the remote object is
    # compatible and how much memory to allocate before bulk data arrives.
    dist.send(descriptor, dst=1)
    transfer = dist.isend(payload, dst=1)

    # The producer must keep the source KV alive until the transfer is complete
    # and the consumer acknowledges the same request/generation.
    transfer.wait()
    completion = torch.empty(3, dtype=torch.int64, device=device)
    dist.recv(completion, src=1)
    release_ok = bool(
        completion.tolist() == [request_id, generation, 1]
    )
    if not release_ok:
        raise RuntimeError(f"invalid completion acknowledgement: {completion.tolist()}")
    dist.send(torch.tensor([1], dtype=torch.int64, device=device), dst=1)


def consumer(args: argparse.Namespace, device: torch.device) -> None:
    (
        wk,
        wv,
        wq,
        remote_prompt,
        remote_new,
        local_prompt,
        local_new,
        cold_prompt,
    ) = deterministic_model_and_requests(device, args.head_dim)

    descriptor = torch.empty(8, dtype=torch.int64, device=device)
    dist.recv(descriptor, src=0)
    (
        request_id,
        prompt_tokens,
        block_size,
        num_blocks,
        generation,
        model_signature,
        head_dim,
        payload_numel,
    ) = descriptor.tolist()

    expected_signature = 424242
    compatible = (
        request_id == 202
        and block_size == args.block_size
        and head_dim == args.head_dim
        and model_signature == expected_signature
        and num_blocks * block_size == prompt_tokens
    )
    if not compatible:
        raise RuntimeError(f"incompatible remote KV descriptor: {descriptor.tolist()}")

    # Scheduler-side block allocation: logical remote blocks are assigned to
    # physical slots before the async data plane fills them.
    physical_table = [7, 3][:num_blocks]
    if max(physical_table) >= args.pool_blocks:
        raise RuntimeError("teaching physical block table exceeds pool")
    kv_pool = torch.zeros(
        args.pool_blocks,
        args.block_size,
        2,
        args.head_dim,
        device=device,
    )
    recv_payload = torch.empty(payload_numel, device=device)
    transfer = dist.irecv(recv_payload, src=0)

    states = {
        201: "RUNNING_LOCAL_KV",
        202: "WAITING_EXTERNAL_KV",
        203: "WAITING_PREFILL",
    }
    budget = args.token_budget

    # Request 201: local prefix hit. Decode one token immediately.
    local_k, local_v = make_kv(local_prompt, wk, wv)
    _local_out = decode_one(local_new, local_k, local_v, wq)
    budget -= 1
    states[201] = "DECODED"

    # Request 203: cold miss. Spend four model-compute tokens on local prefill
    # while request 202's external KV is still transferring.
    cold_tokens = cold_prompt.shape[0]
    if cold_tokens > budget:
        raise RuntimeError("token budget too small for teaching schedule")
    _cold_k, _cold_v = make_kv(cold_prompt, wk, wv)
    budget -= cold_tokens
    states[203] = "PREFILL_READY"

    # Only now does the consumer need request 202. External loading itself did
    # not consume model-compute token budget, but decode cannot start until data
    # and metadata compatibility are complete.
    transfer.wait()
    recv_blocks = recv_payload.view(
        num_blocks, args.block_size, 2, args.head_dim
    )
    for logical_idx, physical_idx in enumerate(physical_table):
        kv_pool[physical_idx].copy_(recv_blocks[logical_idx])

    logical_blocks = torch.stack([kv_pool[p] for p in physical_table], dim=0)
    logical_tokens = logical_blocks.view(-1, 2, args.head_dim)[:prompt_tokens]
    remote_k = logical_tokens[:, 0, :]
    remote_v = logical_tokens[:, 1, :]
    states[202] = "RUNNING_REMOTE_KV"

    if budget < 1:
        raise RuntimeError("no token budget left for remote decode")
    remote_out = decode_one(remote_new, remote_k, remote_v, wq)
    budget -= 1
    states[202] = "DECODED"

    # Correctness-only reference: a production decode worker would not recompute
    # this prompt; we do it here solely to verify handoff semantics.
    ref_k, ref_v = make_kv(remote_prompt, wk, wv)
    ref_out = decode_one(remote_new, ref_k, ref_v, wq)
    err = (remote_out - ref_out).abs().max()

    completion = torch.tensor(
        [request_id, generation, 1], dtype=torch.int64, device=device
    )
    dist.send(completion, dst=0)
    release = torch.empty(1, dtype=torch.int64, device=device)
    dist.recv(release, src=0)
    lifetime_ok = release.item() == 1

    print("=== Mini Inference Engine Capstone ===")
    print(f"backend={dist.get_backend()} producer=rank0 consumer=rank1")
    print(f"token budget       : {args.token_budget} -> used {args.token_budget - budget} -> remaining {budget}")
    print("request 201        : local KV hit -> decode")
    print("request 202        : external descriptor -> physical blocks -> async load -> decode")
    print("request 203        : cold miss -> local prefill while remote KV is in flight")
    print(f"remote block table : logical [0..{num_blocks - 1}] -> physical {physical_table}")
    print(f"descriptor valid   : {compatible}")
    print(f"completion release : {lifetime_ok}")
    print(f"final states       : {states}")
    print(f"max remote decode reference error: {err.item():.3e}")
    print("teaching note      : request states, descriptor fields, and block IDs are a reference model, not exact vLLM internals")
    print("PASS" if lifetime_ok and budget == 0 and err.item() < 1e-5 else "FAIL")


def main() -> None:
    args = parse_args()
    rank, device = init_dist()
    if rank == 0:
        producer(args, device)
    else:
        consumer(args, device)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
