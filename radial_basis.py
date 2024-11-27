import torch


# adapted from e3x: https://e3x.readthedocs.io/stable/_autosummary/e3x.nn.functions.window.triangular_window.html#e3x.nn.functions.window.triangular_window
def triangular_window(
    x: torch.Tensor,
    num: int,  # number of scalars to return
    limit: float = 1.0,
) -> torch.Tensor:
    r"""Triangular window basis functions.

    Computes the basis functions

    .. math::
      \mathrm{triangular\_window}_k(x) = \max\left(
        \min\left(\frac{K}{l}x - k - 1, \frac{K}{l}x + k + 1\right), 0\right)

    where :math:`k=0 \dots K-1` with :math:`K = \text{num}` and
    :math:`l = \text{limit}`.

    Args:
        x: Input tensor.
        num: Number of basis functions :math:`K`.
        limit: Basis functions are distributed between 0 and `limit`.

    Returns:
        Tensor of shape `x.shape + (num,)` containing the values of all basis functions for all values in `x`.
    """
    limit = torch.as_tensor(limit, dtype=x.dtype, device=x.device)
    width = limit / num
    center = limit * torch.arange(num, device=x.device, dtype=x.dtype) / num
    lower = center - width
    upper = center + width

    x_1 = x.unsqueeze(-1)
    temp = torch.minimum((x_1 - lower) / width, -(x_1 - upper) / width)
    return torch.maximum(temp, torch.tensor(0.0, dtype=x.dtype, device=x.device))
