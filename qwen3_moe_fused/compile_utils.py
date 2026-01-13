import torch
from torch import nn


def compile_layers(model: nn.Module) -> None:
    for m in model.base_model.model.model.layers:
        m.forward = torch.compile(m.forward, fullgraph=False, mode="max-autotune-no-cudagraphs")
