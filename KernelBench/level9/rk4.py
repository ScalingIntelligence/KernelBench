import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A model that performs 3D heat diffusion using a 9-point stencil
    and 4th-order Runge-Kutta (RK4) time integration.
    
    We solve: u_t = alpha * Laplacian(u)
    
    The Laplacian is computed with a 1D 9-point (radius-4) stencil in x, y, and z.
    We only update the interior points in each dim, leaving the boundary values unchanged.
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
        Performs 3D heat diffusion simulation using RK4 time integration.
        
        Args:
            u0: Initial 3D field tensor of shape [grid_size, grid_size, grid_size]
        
        Returns:
            Final field after n_steps RK4 updates of the 3D heat equation
        """
        # 3D 8th-order 2nd-derivative Laplacian coefficients
        c0 = -205.0 / 72.0
        c1 = 8.0 / 5.0
        c2 = -1.0 / 5.0
        c3 = 8.0 / 315.0
        c4 = -1.0 / 560.0

        # CFL stability (same constant, but now for RK4)
        c = 0.05

        # Current field
        u = u0.clone()
        device, dtype = u.device, u.dtype

        # Move scalars to same device/dtype as u
        alpha = torch.as_tensor(self.alpha, device=device, dtype=dtype)
        hx = torch.as_tensor(self.hx, device=device, dtype=dtype)
        hy = torch.as_tensor(self.hy, device=device, dtype=dtype)
        hz = torch.as_tensor(self.hz, device=device, dtype=dtype)

        inv_hx2 = 1.0 / (hx * hx)
        inv_hy2 = 1.0 / (hy * hy)
        inv_hz2 = 1.0 / (hz * hz)

        S = inv_hx2 + inv_hy2 + inv_hz2
        dt = c / (alpha * S)

        # Radius of stencil
        r = 4

        # Interior slices (note that boundary values are not updated)
        zc = slice(r, -r)
        yc = slice(r, -r)
        xc = slice(r, -r)

        # Helper to compute Laplacian(u) on interior region
        def laplacian_8th(u_field: torch.Tensor) -> torch.Tensor:
            uc = u_field[zc, yc, xc]

            # x-direction second derivative
            u_xx = (
                c0 * uc
                + c1 * (u_field[zc, yc, r + 1 : -r + 1] + u_field[zc, yc, r - 1 : -r - 1])
                + c2 * (u_field[zc, yc, r + 2 : -r + 2] + u_field[zc, yc, r - 2 : -r - 2])
                + c3 * (u_field[zc, yc, r + 3 : -r + 3] + u_field[zc, yc, r - 3 : -r - 3])
                + c4 * (u_field[zc, yc, r + 4 :] + u_field[zc, yc, : -r - 4])
            ) * inv_hx2

            # y-direction second derivative
            u_yy = (
                c0 * uc
                + c1 * (u_field[zc, r + 1 : -r + 1, xc] + u_field[zc, r - 1 : -r - 1, xc])
                + c2 * (u_field[zc, r + 2 : -r + 2, xc] + u_field[zc, r - 2 : -r - 2, xc])
                + c3 * (u_field[zc, r + 3 : -r + 3, xc] + u_field[zc, r - 3 : -r - 3, xc])
                + c4 * (u_field[zc, r + 4 :, xc] + u_field[zc, : -r - 4, xc])
            ) * inv_hy2

            # z-direction second derivative
            u_zz = (
                c0 * uc
                + c1 * (u_field[r + 1 : -r + 1, yc, xc] + u_field[r - 1 : -r - 1, yc, xc])
                + c2 * (u_field[r + 2 : -r + 2, yc, xc] + u_field[r - 2 : -r - 2, yc, xc])
                + c3 * (u_field[r + 3 : -r + 3, yc, xc] + u_field[r - 3 : -r - 3, yc, xc])
                + c4 * (u_field[r + 4 :, yc, xc] + u_field[: -r - 4, yc, xc])
            ) * inv_hz2

            return u_xx + u_yy + u_zz

        # Workspace for next solution and intermediate stage field
        f = torch.empty_like(u)  # u_{n+1} each step
        u_stage = torch.empty_like(u)  # u_n + a*dt*k
        # Stage vectors (interior only)
        uc_shape = u[zc, yc, xc].shape
        k1 = torch.empty(uc_shape, device=device, dtype=dtype)
        k2 = torch.empty(uc_shape, device=device, dtype=dtype)
        k3 = torch.empty(uc_shape, device=device, dtype=dtype)
        k4 = torch.empty(uc_shape, device=device, dtype=dtype)

        for _ in range(self.n_steps):
            # Interior view of current u
            uc = u[zc, yc, xc]

            # ---- Stage 1: k1 = alpha * Lap(u_n), u_stage = u_n + dt/2 * k1 ----
            lap1 = laplacian_8th(u)
            k1.copy_(alpha * lap1)

            u_stage.copy_(u)  # start from u so boundaries are preserved
            u_stage[zc, yc, xc] = uc + 0.5 * dt * k1

            # ---- Stage 2: k2 = alpha * Lap(u_stage), u_stage = u_n + dt/2 * k2 ----
            lap2 = laplacian_8th(u_stage)
            k2.copy_(alpha * lap2)

            u_stage.copy_(u)  # reset from u again
            u_stage[zc, yc, xc] = uc + 0.5 * dt * k2

            # ---- Stage 3: k3 = alpha * Lap(u_stage), u_stage = u_n + dt * k3 ----
            lap3 = laplacian_8th(u_stage)
            k3.copy_(alpha * lap3)

            u_stage.copy_(u)
            u_stage[zc, yc, xc] = uc + dt * k3

            # ---- Stage 4: k4 = alpha * Lap(u_stage) ----
            lap4 = laplacian_8th(u_stage)
            k4.copy_(alpha * lap4)

            # ---- Final RK4 combination on interior ----
            f.copy_(u)  # boundaries unchanged
            f[zc, yc, xc] = uc + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # Swap for next step
            u, f = f, u

        return u


# Problem configuration
grid_size = 64
n_steps = 10


def get_inputs():
    # Generate input field: [grid_size, grid_size, grid_size]
    # Distribution: Standard normal (mean=0, std=1) via torch.randn()
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
