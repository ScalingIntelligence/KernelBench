import torch


def l2normalize(
        tensor: torch.Tensor, axis: int = -1, eps: float = 1e-8
) -> torch.Tensor:
    """Computes L2 normalization of a tensor."""
    return tensor / (torch.linalg.norm(tensor, ord=2, dim=axis, keepdim=True) + eps)


class Model(torch.nn.Module):
    """HyperEmbedder-inspired reference used in the Kayvon RL kernel."""

    def forward(
        self,
        x: torch.Tensor,
        c_shift: float,
        W: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        new_axis = torch.full((*x.shape[:-1], 1), c_shift, device=x.device, dtype=x.dtype)
        concatenated = torch.cat([x, new_axis], dim=-1)

        # l2 norm
        normalized = l2normalize(concatenated, axis=-1)

        # Linear layer followed by scaler
        projected = torch.matmul(normalized, W)
        scaled = projected * scale

        # l2 norm
        return l2normalize(scaled, axis=-1)


# Problem configuration
batch_size = 64
in_features = 128
hidden_size = 256


def get_inputs():
    x = torch.randn(batch_size, in_features)
    W = torch.randn(in_features + 1, hidden_size)
    scale = torch.randn(hidden_size)
    c_shift = torch.randn(1).item()
    return [x, c_shift, W, scale]


def get_init_inputs():
    return []
