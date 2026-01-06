import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A model that performs 3D heat diffusion using a 9-point stencil
    and explicit Euler time integration.
    
    To compute the next time step: u^{n+1} = u^n + dt * alpha * Laplacian(u^n)
    
    The Laplacian is computed with a 1D 9-point (4 neighbors on each side) stencil
    in x, y, and z. We only update the interior points in each dim, leaving the
    boundary values unchanged.
    """

    def __init__(self, alpha: float, hx: float, hy: float, hz: float, n_steps: int):
        super(Model, self).__init__()
        self.alpha = alpha
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.n_steps = n_steps

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        """
        Performs 3D heat diffusion simulation.
        
        Args:
            u0: Initial 3D field tensor of shape [grid_size, grid_size, grid_size]
        
        Returns:
            Final field after n_steps Euler updates of 3D heat equation
        """
        # 3D 8th-order 2nd-derivative Laplacian coefficients
        c0 = -205.0 / 72.0
        c1 = 8.0 / 5.0
        c2 = -1.0 / 5.0
        c3 = 8.0 / 315.0
        c4 = -1.0 / 560.0

        # CFL stability
        c = 0.05

        # Move scalars to same device/dtype as u
        u = u0.clone()
        device, dtype = u.device, u.dtype
        alpha = torch.as_tensor(self.alpha, device=device, dtype=dtype)
        hx = torch.as_tensor(self.hx, device=device, dtype=dtype)
        hy = torch.as_tensor(self.hy, device=device, dtype=dtype)
        hz = torch.as_tensor(self.hz, device=device, dtype=dtype)

        inv_hx2 = 1.0 / (hx * hx)
        inv_hy2 = 1.0 / (hy * hy)
        inv_hz2 = 1.0 / (hz * hz)

        S = inv_hx2 + inv_hy2 + inv_hz2
        dt = c / (alpha * S)

        f = torch.empty_like(u)

        # Radius of stencil
        r = 4
        # Interior slices (these stay fixed each step)
        zc = slice(r, -r)
        yc = slice(r, -r)
        xc = slice(r, -r)

        for _ in range(self.n_steps):
            # copy old solution; boundaries remain unchanged
            f.copy_(u)

            # center region
            uc = u[zc, yc, xc]

            # x-direction second derivative
            u_xx = (
                c0 * uc
                + c1 * (u[zc, yc, r + 1 : -r + 1] + u[zc, yc, r - 1 : -r - 1])
                + c2 * (u[zc, yc, r + 2 : -r + 2] + u[zc, yc, r - 2 : -r - 2])
                + c3 * (u[zc, yc, r + 3 : -r + 3] + u[zc, yc, r - 3 : -r - 3])
                + c4 * (u[zc, yc, r + 4 :] + u[zc, yc, : -r - 4])
            ) * inv_hx2

            # y-direction second derivative
            u_yy = (
                c0 * uc
                + c1 * (u[zc, r + 1 : -r + 1, xc] + u[zc, r - 1 : -r - 1, xc])
                + c2 * (u[zc, r + 2 : -r + 2, xc] + u[zc, r - 2 : -r - 2, xc])
                + c3 * (u[zc, r + 3 : -r + 3, xc] + u[zc, r - 3 : -r - 3, xc])
                + c4 * (u[zc, r + 4 :, xc] + u[zc, : -r - 4, xc])
            ) * inv_hy2

            # z-direction second derivative
            u_zz = (
                c0 * uc
                + c1 * (u[r + 1 : -r + 1, yc, xc] + u[r - 1 : -r - 1, yc, xc])
                + c2 * (u[r + 2 : -r + 2, yc, xc] + u[r - 2 : -r - 2, yc, xc])
                + c3 * (u[r + 3 : -r + 3, yc, xc] + u[r - 3 : -r - 3, yc, xc])
                + c4 * (u[r + 4 :, yc, xc] + u[: -r - 4, yc, xc])
            ) * inv_hz2

            lap = u_xx + u_yy + u_zz

            # Explicit Euler update on interior only
            f[zc, yc, xc] = uc + dt * alpha * lap

            # swap
            u, f = f, u

        return u


# Problem configuration
grid_size = 64
n_steps = 10


def get_inputs():
    # Generate random 3D initial field: [grid_size, grid_size, grid_size]
    u0 = torch.randn(grid_size, grid_size, grid_size, dtype=torch.float32).contiguous()
    return [u0]


def get_init_inputs():
    # Random diffusion coefficient alpha in [0.1, 5.0]
    alpha = torch.rand(1).item() * 4.9 + 0.1
    
    # Random grid spacings hx, hy, hz in [0.5, 2.0]
    hx = torch.rand(1).item() * 1.5 + 0.5
    hy = torch.rand(1).item() * 1.5 + 0.5
    hz = torch.rand(1).item() * 1.5 + 0.5
    
    return [alpha, hx, hy, hz, n_steps]
