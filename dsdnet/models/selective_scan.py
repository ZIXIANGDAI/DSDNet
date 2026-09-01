"""Selective scan and Gilbert traversal utilities.

Implementation patterns are adapted from Mamba, VMamba, and gilbert. See
THIRD_PARTY_NOTICES.md for source links and license attribution.
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import selective_scan_cuda
except ImportError:
    selective_scan_cuda = None
    warnings.warn("selective_scan_cuda is unavailable; using the PyTorch fallback.")


def _sign(value):
    return -1 if value < 0 else int(value > 0)


def _gilbert_generate_2d(x, y, ax, ay, bx, by):
    width, height = abs(ax + ay), abs(bx + by)
    dax, day, dbx, dby = _sign(ax), _sign(ay), _sign(bx), _sign(by)
    if height == 1:
        for _ in range(width):
            yield x, y
            x, y = x + dax, y + day
        return
    if width == 1:
        for _ in range(height):
            yield x, y
            x, y = x + dbx, y + dby
        return
    ax2, ay2, bx2, by2 = ax // 2, ay // 2, bx // 2, by // 2
    width2, height2 = abs(ax2 + ay2), abs(bx2 + by2)
    if 2 * width > 3 * height:
        if width2 % 2 and width > 2:
            ax2, ay2 = ax2 + dax, ay2 + day
        yield from _gilbert_generate_2d(x, y, ax2, ay2, bx, by)
        yield from _gilbert_generate_2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
    else:
        if height2 % 2 and height > 2:
            bx2, by2 = bx2 + dbx, by2 + dby
        yield from _gilbert_generate_2d(x, y, bx2, by2, ax2, ay2)
        yield from _gilbert_generate_2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
        yield from _gilbert_generate_2d(
            x + ax - dax + bx2 - dbx,
            y + ay - day + by2 - dby,
            -bx2,
            -by2,
            -(ax - ax2),
            -(ay - ay2),
        )


def gilbert_2d(width, height):
    yield from _gilbert_generate_2d(0, 0, width, 0, 0, height)


class ChannelFirstLinear(nn.Linear):
    def __init__(self, *args, groups=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups

    def forward(self, feature):
        return F.conv1d(feature, self.weight[:, :, None], self.bias, groups=self.groups)


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = probability

    def forward(self, feature):
        if self.probability == 0.0 or not self.training:
            return feature
        keep = 1.0 - self.probability
        shape = (feature.shape[0],) + (1,) * (feature.ndim - 1)
        mask = (keep + torch.rand(shape, dtype=feature.dtype, device=feature.device)).floor()
        return feature * mask / keep


def selective_scan_torch(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    batch, groups, state_size, length = B.shape
    channels_per_group = u.shape[1] // groups
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B[:, :, None].expand(-1, -1, channels_per_group, -1, -1).reshape(batch, -1, state_size, length)
    C = C[:, :, None].expand(-1, -1, channels_per_group, -1, -1).reshape(batch, -1, state_size, length)
    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    delta_b_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    state = A.new_zeros((batch, u.shape[1], state_size))
    outputs = []
    for index in range(length):
        state = delta_a[:, :, index] * state + delta_b_u[:, :, index]
        outputs.append(torch.einsum("bdn,bdn->bd", state, C[:, :, :, index]))
    output = torch.stack(outputs, dim=2)
    return output if D is None else output + u * D.unsqueeze(-1)


class _SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, delta_bias):
        output, state, *_ = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, True)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, state)
        return output

    @staticmethod
    def backward(ctx, grad):
        u, delta, A, B, C, D, delta_bias, state = ctx.saved_tensors
        if grad.stride(-1) != 1:
            grad = grad.contiguous()
        values = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, grad, state, None, None, True, False
        )
        return (*values[:7],)


def selective_scan(u, delta, A, B, C, D, delta_bias):
    if selective_scan_cuda is not None and u.is_cuda:
        return _SelectiveScanCuda.apply(u, delta, A, B, C, D, delta_bias)
    return selective_scan_torch(u, delta, A, B, C, D, delta_bias)


def initialize_mamba(d_state, dt_rank, d_inner, groups=1):
    dt_layers = []
    for _ in range(groups):
        layer = nn.Linear(dt_rank, d_inner, bias=True)
        std = dt_rank ** -0.5
        nn.init.uniform_(layer.weight, -std, std)
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            layer.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        dt_layers.append(layer)
    weights = nn.Parameter(torch.stack([layer.weight for layer in dt_layers]))
    biases = nn.Parameter(torch.stack([layer.bias for layer in dt_layers]))
    base = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1).repeat(d_inner, 1)
    a_logs = nn.Parameter(torch.log(base)[None].repeat(groups, 1, 1).flatten(0, 1))
    ds = nn.Parameter(torch.ones(groups * d_inner))
    return a_logs, ds, weights, biases
