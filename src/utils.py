import random
import numpy as np
import torch


def resolve_device(name: str = "auto") -> str:
    """Resolve an explicit accelerator or select CUDA, then MPS, then CPU."""
    cuda = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps = mps_backend is not None and mps_backend.is_available()
    if name == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device: {name}")
    if (name == "cuda" and not cuda) or (name == "mps" and not mps):
        raise ValueError(f"Requested device {name!r} is unavailable; use --device cpu or auto.")
    return name


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        torch.mps.manual_seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
