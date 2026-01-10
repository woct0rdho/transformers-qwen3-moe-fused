import torch


def get_rtol_atol(actual, expect):
    actual = actual.float()
    expect = expect.float()
    diff = (actual - expect).abs()
    eps = torch.tensor(torch.finfo(actual.dtype).eps, device=actual.device, dtype=actual.dtype)
    rdiff = diff / torch.maximum(torch.maximum(actual.abs(), expect.abs()), eps)
    return (
        f"mean_rtol={rdiff.mean().item():.3g} "
        f"max_rtol={rdiff.max().item():.3g} "
        f"mean_atol={diff.max().item():.3g} "
        f"max_atol={diff.max().item():.3g}"
    )
