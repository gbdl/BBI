"""
Functions that compute derivatives for ODEs and PDEs
"""

import torch


def diff_single(y, x, device):
    """Computes the derivative of a single y w.r.t. x
    """

    yp = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y, device=device),
        create_graph=True,
        retain_graph=True,
    )[0]
    return yp.transpose(0, 1)


def diff_many(y, x, device):
    """Computes the derivative of many y's w.r.t. x
    """
    return [diff_single(y[i], x, device) for i in range(y.shape[0])]
