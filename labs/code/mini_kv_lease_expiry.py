#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

HEARTBEAT = 1
COMPLETE = 2


@dataclass
class LeaseRecord:
    request_id: int
    generation: int
    expiry: float
    released: bool = False
    release_reason: str | None = None

    def heartbeat(self, generation: int, now: float, extension: float) -> bool:
        """Renew only the allocation generation the heartbeat was issued for."""
        if self.released or generation != self.generation or now >= self.expiry:
            return False
        self.expiry = max(self.expiry, now + extension)
        return True

    def complete(self, generation: int, now: float) -> bool:
        if self.released or generation != self.generation or now >= self.expiry:
            return False
        self.released = True
        self.release_reason = "completion"
        return True

    def expire(self, now: float) -> bool:
        if not self.released and now >= self.expiry:
            self.released = True
            self.release_reason = "expiry"
            return True
        return False


def init_dist() -> tuple[int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Launch with: torchrun --standalone --nproc-per-node=2 "
            "labs/code/mini_kv_lease_expiry.py"
        )
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise ValueError("This lab uses exactly two ranks: producer and consumer")

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return rank, torch.device("cuda", local_rank)
    return rank, torch.device("cpu")


def send_event(device: torch.device, event: int, request_id: int, generation: int, now: float) -> None:
    packet = torch.tensor(
        [event, request_id, generation, int(round(now * 1000))],
        dtype=torch.int64,
        device=device,
    )
    dist.send(packet, dst=0)


def recv_event(device: torch.device) -> tuple[int, int, int, float]:
    packet = torch.empty(4, dtype=torch.int64, device=device)
    dist.recv(packet, src=1)
    event, request_id, generation, now_ms = [int(x) for x in packet.tolist()]
    return event, request_id, generation, now_ms / 1000.0


def main() -> None:
    rank, device = init_dist()

    # Virtual time keeps the lab deterministic and instant. The ratios mirror the
    # current NIXL defaults at review time, scaled down for readability:
    # lease 30s -> 6 units, heartbeat 5s -> 1 unit, extension 20s -> 4 units.
    lease_duration = 6.0
    lease_extension = 4.0

    if rank == 1:
        # Request 501 stays alive, then explicitly completes.
        send_event(device, HEARTBEAT, 501, 1, 3.0)
        # A stale heartbeat for an old generation must not renew reused memory.
        send_event(device, HEARTBEAT, 501, 0, 4.0)
        send_event(device, COMPLETE, 501, 1, 5.0)

        summary = torch.empty(5, dtype=torch.float64, device=device)
        dist.recv(summary, src=0)
        ok = bool(summary[0].item())
        if not ok:
            raise RuntimeError("producer-side lease state-machine validation failed")
    else:
        records = {
            501: LeaseRecord(501, generation=1, expiry=lease_duration),
            502: LeaseRecord(502, generation=1, expiry=lease_duration),
        }

        event, req, generation, now = recv_event(device)
        assert event == HEARTBEAT and req == 501
        renewed = records[req].heartbeat(generation, now, lease_extension)
        renewed_expiry = records[501].expiry

        event, req, generation, now = recv_event(device)
        assert event == HEARTBEAT and req == 501
        stale_ignored = not records[req].heartbeat(generation, now, lease_extension)
        expiry_after_stale = records[501].expiry

        event, req, generation, now = recv_event(device)
        assert event == COMPLETE and req == 501
        completion_released = records[req].complete(generation, now)

        # Request 502 never sends heartbeat/completion. The timeout is the leak-safe fallback.
        timeout_now = 6.5
        timeout_released = records[502].expire(timeout_now)

        ok = all(
            [
                renewed,
                abs(renewed_expiry - 7.0) < 1e-9,
                stale_ignored,
                abs(expiry_after_stale - 7.0) < 1e-9,
                completion_released,
                records[501].release_reason == "completion",
                timeout_released,
                records[502].release_reason == "expiry",
            ]
        )

        print("=== Mini KV Lease / Expiry State Machine ===")
        print(f"backend={dist.get_backend()} device={device.type}")
        print(f"initial lease expiry          : 6.0")
        print(f"heartbeat req501 @ t=3.0      : renewed -> {renewed_expiry:.1f}")
        print(f"stale generation heartbeat    : {'ignored' if stale_ignored else 'accepted'}")
        print(f"completion req501 @ t=5.0     : release={records[501].release_reason}")
        print(f"silent req502 @ t={timeout_now:.1f}        : release={records[502].release_reason}")
        print("NOTE: virtual time + toy generation IDs teach lifetime safety; this is not an exact vLLM lease implementation.")
        print("PASS" if ok else "FAIL")

        summary = torch.tensor(
            [
                float(ok),
                renewed_expiry,
                expiry_after_stale,
                float(completion_released),
                float(timeout_released),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.send(summary, dst=1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
