#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser(description='Toy allocation-vs-transfer-region descriptor lab before real NIXL registration.')
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--block-size', type=int, default=4)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--head-dim', type=int, default=8)
    p.add_argument('--block-id', type=int, default=2)
    return p.parse_args()


def init_dist():
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        raise RuntimeError('Launch with torchrun --standalone --nproc-per-node=2 ...')
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 2:
        raise ValueError('This lab uses 2 ranks: producer and consumer')
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world
    dist.init_process_group('nccl' if use_cuda else 'gloo')
    if use_cuda:
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    return rank, device


def build_pool(args, device):
    # K and V are packed in the last dimension for a compact teaching layout.
    shape = (args.layers, args.blocks, args.block_size, args.heads, 2 * args.head_dim)
    torch.manual_seed(4101)
    return torch.randn(shape, device=device)


def describe(pool: torch.Tensor):
    storage_base = pool.untyped_storage().data_ptr()
    storage_bytes = pool.untyped_storage().nbytes()
    element_size = pool.element_size()
    regions = []
    for layer in range(pool.shape[0]):
        cache = pool[layer]
        region_base = cache.data_ptr()
        block_stride = cache.stride(0) * element_size
        block_len = cache[0].numel() * element_size
        regions.append((region_base - storage_base, block_len, block_stride))
    return storage_base, storage_bytes, regions


def main():
    args = parse_args()
    rank, device = init_dist()
    if not 0 <= args.block_id < args.blocks:
        raise ValueError('--block-id out of range')

    pool = build_pool(args, device)
    storage_base, storage_bytes, regions = describe(pool)
    element_size = pool.element_size()

    header = torch.tensor([
        args.layers,
        args.blocks,
        args.block_size,
        args.heads,
        args.head_dim,
        element_size,
        storage_bytes,
    ], dtype=torch.int64, device=device)
    region_tensor = torch.tensor(regions, dtype=torch.int64, device=device)

    if rank == 0:
        # Remote code cannot dereference storage_base. We send normalized offsets
        # and geometry instead; real NIXL additionally exchanges transport/agent metadata.
        dist.send(header, dst=1)
        dist.send(region_tensor, dst=1)
        for layer in range(args.layers):
            dist.send(pool[layer, args.block_id].contiguous(), dst=1)
        result = torch.empty(3, device=device)
        dist.recv(result, src=1)
    else:
        remote_header = torch.empty_like(header)
        remote_regions = torch.empty_like(region_tensor)
        dist.recv(remote_header, src=0)
        dist.recv(remote_regions, src=0)

        local_base, local_storage_bytes, local_regions = describe(pool)
        local_regions_t = torch.tensor(local_regions, dtype=torch.int64, device=device)
        header_ok = torch.equal(remote_header, header)
        geometry_ok = torch.equal(remote_regions, local_regions_t)

        max_err = torch.tensor(0.0, device=device)
        recv_pool = torch.empty_like(pool)
        for layer in range(args.layers):
            recv_block = torch.empty_like(pool[layer, args.block_id])
            dist.recv(recv_block, src=0)
            recv_pool[layer, args.block_id].copy_(recv_block)
            max_err = torch.maximum(max_err, (recv_block - pool[layer, args.block_id]).abs().max())

        # Prove that the pointer itself is process-local: the consumer validates
        # offsets/strides against its own allocation rather than using producer addresses.
        pointer_is_local = local_base == storage_base  # usually true on CPU forked layout can vary; do not rely on inequality
        _ = pointer_is_local, local_storage_bytes
        result = torch.tensor([float(header_ok), float(geometry_ok), float(max_err)], device=device)
        dist.send(result, dst=0)

    if rank == 0:
        print('=== Mini Registered Region Descriptor ===')
        print(f'backend={dist.get_backend()} device={device.type}')
        print(f'one storage allocation bytes : {storage_bytes}')
        print(f'logical transfer regions     : {len(regions)}')
        print(f'region offsets/len/stride    : {regions}')
        print(f'header compatible            : {bool(result[0].item())}')
        print(f'region geometry compatible   : {bool(result[1].item())}')
        print(f'max transferred-block error : {result[2].item():.3e}')
        print('NOTE: data_ptr/offset metadata is teaching-only; no NIXL register_memory, rkey, UCX, or RDMA is used.')
        print('PASS' if result[0].item() == 1 and result[1].item() == 1 and result[2].item() < 1e-6 else 'FAIL')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
