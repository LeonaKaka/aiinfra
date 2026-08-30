#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


PROTOCOL_VERSION = 1
REGION_ID = 7
REQUEST_ID = 501


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Toy KV control-plane handshake + completion/lifetime protocol before real NIXL registration."
    )
    p.add_argument("--tokens", type=int, default=12)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=8)
    return p.parse_args()


def init_dist():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch with torchrun --standalone --nproc-per-node=2 ...")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("This lab intentionally uses 2 ranks: producer and consumer")
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, device


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    score = torch.einsum("hd,thd->ht", q, k) * scale
    prob = torch.softmax(score, dim=-1)
    return torch.einsum("ht,thd->hd", prob, v)


def main():
    args = parse_args()
    rank, device = init_dist()
    generation = 1

    torch.manual_seed(2101)
    prompt = torch.randn(args.tokens, args.heads, args.head_dim, device=device)
    wk = torch.randn(args.head_dim, args.head_dim, device=device) / args.head_dim**0.5
    wv = torch.randn(args.head_dim, args.head_dim, device=device) / args.head_dim**0.5
    q = torch.randn(args.heads, args.head_dim, device=device)
    k = torch.einsum("thd,df->thf", prompt, wk).contiguous()
    v = torch.einsum("thd,df->thf", prompt, wv).contiguous()
    reference = attention(q, k, v)

    # This small tensor is a toy control-plane descriptor. It is NOT an RDMA
    # memory descriptor or an rkey. Fields:
    # version, request, region, generation, tokens, heads, head_dim, bytes(K+V)
    descriptor = torch.tensor(
        [
            PROTOCOL_VERSION,
            REQUEST_ID,
            REGION_ID,
            generation,
            args.tokens,
            args.heads,
            args.head_dim,
            (k.numel() + v.numel()) * k.element_size(),
        ],
        dtype=torch.int64,
        device=device,
    )

    if rank == 0:
        dist.send(descriptor, dst=1)

        accept = torch.empty(4, dtype=torch.int64, device=device)
        dist.recv(accept, src=1)
        accepted = bool(accept[3].item())
        if not accepted:
            raise RuntimeError("consumer rejected descriptor")

        work_k = dist.isend(k, dst=1)
        work_v = dist.isend(v, dst=1)
        work_k.wait()
        work_v.wait()

        # Region must stay logically owned by this generation until consumer
        # reports that it is done. Real systems need an equivalent lifetime rule.
        completion = torch.empty(4, dtype=torch.int64, device=device)
        dist.recv(completion, src=1)
        completion_ok = (
            completion[0].item() == REQUEST_ID
            and completion[1].item() == REGION_ID
            and completion[2].item() == generation
            and completion[3].item() == 1
        )
        next_generation = generation + 1 if completion_ok else generation
        summary = torch.tensor([int(accepted), int(completion_ok), next_generation], device=device)
    else:
        incoming = torch.empty_like(descriptor)
        dist.recv(incoming, src=0)
        version, request_id, region_id, remote_gen, tokens, heads, head_dim, nbytes = [
            int(x) for x in incoming.tolist()
        ]

        expected_bytes = 2 * tokens * heads * head_dim * torch.empty((), dtype=torch.float32).element_size()
        descriptor_valid = (
            version == PROTOCOL_VERSION
            and request_id == REQUEST_ID
            and region_id == REGION_ID
            and tokens == args.tokens
            and heads == args.heads
            and head_dim == args.head_dim
            and nbytes == expected_bytes
        )

        accept = torch.tensor(
            [request_id, region_id, remote_gen, int(descriptor_valid)],
            dtype=torch.int64,
            device=device,
        )
        dist.send(accept, dst=0)
        if not descriptor_valid:
            raise RuntimeError("descriptor validation failed")

        recv_k = torch.empty(tokens, heads, head_dim, device=device)
        recv_v = torch.empty_like(recv_k)
        wk_recv = dist.irecv(recv_k, src=0)
        wv_recv = dist.irecv(recv_v, src=0)
        wk_recv.wait()
        wv_recv.wait()

        out = attention(q, recv_k, recv_v)
        err = (out - reference).abs().max()

        completion = torch.tensor(
            [request_id, region_id, remote_gen, 1], dtype=torch.int64, device=device
        )
        dist.send(completion, dst=0)
        summary = torch.tensor([int(descriptor_valid), 1, remote_gen + 1], device=device)

    if rank == 1:
        error = err.detach()
    else:
        error = torch.tensor(0.0, device=device)
    dist.broadcast(error, src=1)
    dist.broadcast(summary, src=0)

    if rank == 0:
        print("=== Mini KV Handshake + Lifetime ===")
        print(f"backend={dist.get_backend()} device={device.type}")
        print(f"descriptor accepted      : {bool(summary[0].item())}")
        print(f"completion acknowledged  : {bool(summary[1].item())}")
        print(f"region generation        : {generation} -> {int(summary[2].item())}")
        print(f"max attention error      : {error.item():.3e}")
        print("NOTE: region_id/generation are a teaching protocol, not NIXL memory registration/rkeys.")
        print("PASS" if summary[:2].all().item() and error.item() < 1e-5 else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
