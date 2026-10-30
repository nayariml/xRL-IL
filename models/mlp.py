from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from xRL.utils.torch_util import SquashedNormal


def _normalize_dim(dim: Union[int, Sequence[int]]):
    if isinstance(dim, Sequence):
        return int(dim[0])
    return int(dim)


def get_activation(act_name: str):
    name = act_name.lower()

    mapping: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "gelutanh": lambda: nn.GELU(approximate="tanh"),
        "leakyrelu": nn.LeakyReLU,
        "linear": nn.Identity,
        "lrelu": nn.LeakyReLU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "silu": nn.SiLU,
        "sigmoid": nn.Sigmoid,
        "softmax": lambda: nn.Softmax(dim=-1),
        "tanh": nn.Tanh,
    }

    if name not in mapping:
        raise ValueError(f"Invalid activation function: {act_name!r}")

    return mapping[name]


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_layers: Optional[Sequence[int]] = None,
    act: str = "relu",
    output_activation: Optional[str] = None):
    in_dim = _normalize_dim(in_dim)
    hidden_layers = list(hidden_layers) if hidden_layers is not None else [512, 256, 128]

    act_cls = get_activation(act)
    out_act_cls = get_activation(output_activation) if output_activation is not None else None

    layer_dims = [in_dim, *hidden_layers, out_dim]
    layers: list[nn.Module] = []

    for i, (din, dout) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        layers.append(nn.Linear(din, dout))
        is_last = (i == len(layer_dims) - 2)
        if not is_last:
            layers.append(act_cls())
        elif out_act_cls is not None:
            layers.append(out_act_cls())

    return nn.Sequential(*layers)


class MLPBackbone(nn.Module):
    def __init__(
        self,
        in_dim: Union[int, Sequence[int]],
        out_dim: int,
        hidden_layers: Optional[Sequence[int]] = None,
        act: Optional[str] = None,
        output_activation: Optional[str] = None):
        super().__init__()
        if act is None:
            act = "relu"

        self.mlp = build_mlp(
            in_dim=_normalize_dim(in_dim),
            out_dim=out_dim,
            hidden_layers=hidden_layers,
            act=act,
            output_activation=output_activation,
        )

    def forward(self, x: Tensor):
        return self.mlp(x)


class TanhDiagGaussian(MLPBackbone):
    """
    Squashed Gaussian policy: network outputs (mu, log_std),
    then actions are sampled from SquashedNormal(mu, std) via tanh.
    """
    def __init__(
        self,
        obs_dim: Union[int, Sequence[int]],
        act_dim: int,
        hidden_layers: Optional[Sequence[int]] = None,
        act: Optional[str] = None,
        log_std_bounds: Tuple[float, float] = (-20.0, 2.0)):
        super().__init__(
            in_dim=obs_dim,
            out_dim=act_dim * 2,
            hidden_layers=hidden_layers,
            act=act,
        )
        self.log_std_min, self.log_std_max = log_std_bounds

    def _build_distribution(self, state: Tensor):
        mu, log_std = self.mlp(state).chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return SquashedNormal(mu, std)

    def forward(self, state: Tensor, sample: bool = True):
        dist = self._build_distribution(state)
        return dist.rsample() if sample else dist.mean

    def sample_actions_logprob(self, state: Tensor):
        dist = self._build_distribution(state)
        actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        return actions, dist, log_prob


class TwinQ(nn.Module):
    """
    Twin Q-functions Q1(s, a) and Q2(s, a).
    """
    def __init__(
        self,
        obs_dim: Union[int, Sequence[int]],
        act_dim: int,
        hidden_layers: Optional[Sequence[int]] = None,
        act: Optional[str] = None):
        super().__init__()
        s_dim = _normalize_dim(obs_dim)
        self.mlp_q1 = MLPBackbone(in_dim=s_dim + act_dim, out_dim=1, hidden_layers=hidden_layers, act=act)
        self.mlp_q2 = MLPBackbone(in_dim=s_dim + act_dim, out_dim=1, hidden_layers=hidden_layers, act=act)

    def _concat(self, state: Tensor, action: Tensor):
        return torch.cat((state, action), dim=-1)

    def forward_q1_q2(self, state: Tensor, action: Tensor):
        x = self._concat(state, action)
        return self.mlp_q1(x), self.mlp_q2(x)

    def min_q(self, state: Tensor, action: Tensor):
        q1, q2 = self.forward_q1_q2(state, action)
        return torch.min(q1, q2)
