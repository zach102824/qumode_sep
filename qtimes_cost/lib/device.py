"""Device helpers — CUDA when available, else CPU. Dual-GPU for chaodao 2×5090."""
from __future__ import annotations
from typing import List, Optional, Sequence, Union

import torch


def cuda_count() -> int:
    return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0


def resolve_device(spec: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve a device string. Empty/None -> cuda:0 if available else cpu."""
    if spec is None or spec == "":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(spec, torch.device):
        return spec
    s = str(spec).strip()
    if s.isdigit():
        s = f"cuda:{s}"
    dev = torch.device(s)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if dev.type == "cuda" and dev.index is not None and dev.index >= cuda_count():
        raise ValueError(f"Requested {dev} but only {cuda_count()} CUDA device(s)")
    return dev


def dual_devices(gpus: Optional[Sequence[str]] = None) -> List[torch.device]:
    """Return two devices for parallel evolve candidates.

    Default: cuda:0 and cuda:1 when ≥2 GPUs; else [cuda:0,cuda:0] if 1 GPU;
    else [cpu, cpu] for smoke/tests.
    """
    if gpus:
        return [resolve_device(g) for g in list(gpus)[:2]]
    n = cuda_count()
    if n >= 2:
        return [torch.device("cuda:0"), torch.device("cuda:1")]
    if n == 1:
        return [torch.device("cuda:0"), torch.device("cuda:0")]
    return [torch.device("cpu"), torch.device("cpu")]


def parse_gpus_arg(s: str) -> List[str]:
    """'0,1' or 'cuda:0,cuda:1' -> list of device specs."""
    if not s or not str(s).strip():
        return []
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(part if part.startswith("cuda") or part == "cpu" else part)
    return out
