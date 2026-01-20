import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    MesaNet (Sequence Modeling by Locally Optimal Test-Time Training) - Reference implementation.
    
    MesaNet uses "test-time training" via Conjugate Gradient (CG) to solve a linear system:
        (H_kk + lambda*I) @ q_star = q
    
    Where:
    - H_kk is a recurrently updated state matrix: h_kk[t] = h_kk[t-1] * exp(g[t]) + outer(k[t]*beta[t], k[t])
    - H_kv is another state matrix: h_kv[t] = h_kv[t-1] * exp(g[t]) + outer(k[t]*beta[t], v[t])
    - lambda is a regularization parameter (per head, per dimension)
    
    The output is: o = H_kv @ q_star
    
    Key insight: Instead of materializing the full attention matrix, MesaNet maintains
    compact state matrices and solves for q_star iteratively using CG, which acts as
    a form of "test-time training" that adapts to the current input.
    
    Based on: "MesaNet: Sequence Modeling by Locally Optimal Test-Time Training"
    """
    
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        head_dim: int = 128,
        use_output_gate: bool = False,
        lambda_lower_bound: float = 0.25,
        max_cg_iteration: int = 30,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.key_dim = num_heads * head_dim
        self.value_dim = self.key_dim  # MesaNet uses same dim for V as K
        self.use_output_gate = use_output_gate
        self.lambda_lower_bound = lambda_lower_bound
        self.max_cg_iteration = max_cg_iteration
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        
        # Gate projections
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=True)  # For g (decay)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)  # For beta
        
        # Lambda parameters: regularization per head per dimension
        # Initialized to 1.0, then transformed via softplus + lower_bound
        lambda_initial_value = 1.0
        init_lamb_value = torch.log(torch.exp(torch.tensor(lambda_initial_value - lambda_lower_bound)) - 1.0)
        self.lambda_params = nn.Parameter(torch.empty(self.key_dim, dtype=torch.float32).fill_(init_lamb_value))
        
        # Output gate and projection
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        
        # RMSNorm weight
        self.o_norm_weight = nn.Parameter(torch.ones(self.head_dim))
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask (unused in this reference)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [B, T, key_dim]
        k = self.k_proj(x)  # [B, T, key_dim]
        v = self.v_proj(x)  # [B, T, value_dim]
        
        # Apply SiLU activation (simulating short convolution effect)
        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)
        
        # Reshape to multi-head format
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # L2 normalize Q and K (as done in the kernel)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # Compute beta (sigmoid) and g (log-sigmoid decay)
        beta = torch.sigmoid(self.b_proj(x))  # [B, T, num_heads]
        g = F.logsigmoid(self.a_proj(x))  # [B, T, num_heads] - negative values
        
        # Compute lambda: softplus + lower_bound, reshaped to [num_heads, head_dim]
        lamb = F.softplus(self.lambda_params) + self.lambda_lower_bound
        lamb = lamb.view(self.num_heads, self.head_dim)  # [num_heads, head_dim]
        
        # ============================================
        # MesaNet Core: Test-Time Training via CG
        # ============================================
        o = self._mesa_net_attention(q, k, v, g, beta, lamb)
        
        # Output normalization
        if self.use_output_gate:
            gate = self.g_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            o = self._gated_rms_norm(o, gate, self.o_norm_weight)
        else:
            o = self._rms_norm(o, self.o_norm_weight)
        
        # Reshape and project output
        o = o.view(batch_size, seq_len, self.value_dim)
        o = self.o_proj(o)
        
        return o
    
    def _mesa_net_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        lamb: torch.Tensor
    ) -> torch.Tensor:
        """
        MesaNet attention using Conjugate Gradient solver.
        
        Steps:
        1. Build state matrices h_kk and h_kv recurrently
        2. Solve (h_kk + lambda*I) @ q_star = q using CG
        3. Output: o = h_kv @ q_star
        
        Args:
            q: [B, T, num_heads, head_dim]
            k: [B, T, num_heads, head_dim]
            v: [B, T, num_heads, head_dim]
            g: [B, T, num_heads] - log-sigmoid decay
            beta: [B, T, num_heads] - scaling
            lamb: [num_heads, head_dim] - regularization
            
        Returns:
            o: [B, T, num_heads, head_dim]
        """
        B, T, H, D = q.shape
        
        # Work in float32 for stability
        q = q.float()
        k = k.float()
        v = v.float()
        g = g.float()
        beta = beta.float()
        lamb = lamb.float()
        
        outputs = []
        
        for b in range(B):
            batch_outputs = []
            
            for h in range(H):
                # Initialize state matrices
                h_kk = torch.zeros(D, D, device=q.device, dtype=torch.float32)  # [head_dim, head_dim]
                h_kv = torch.zeros(D, D, device=q.device, dtype=torch.float32)  # [head_dim, head_dim]
                
                # Build state matrices recurrently
                h_kk_all = []
                h_kv_all = []
                
                for t in range(T):
                    k_t = k[b, t, h]  # [D]
                    v_t = v[b, t, h]  # [D]
                    g_t = g[b, t, h]  # scalar
                    beta_t = beta[b, t, h]  # scalar
                    
                    # Update states with exponential decay
                    # h_kk = h_kk * exp(g) + outer(k*beta, k)
                    k_beta = k_t * beta_t
                    h_kk = h_kk * torch.exp(g_t) + torch.outer(k_beta, k_t)
                    h_kv = h_kv * torch.exp(g_t) + torch.outer(k_beta, v_t)
                    
                    h_kk_all.append(h_kk.clone())
                    h_kv_all.append(h_kv.clone())
                
                # Stack states: [T, D, D]
                h_kk_all = torch.stack(h_kk_all, dim=0)
                h_kv_all = torch.stack(h_kv_all, dim=0)
                
                # Get lambda for this head: [D]
                lamb_h = lamb[h]  # [D]
                
                # Solve for each time step using CG
                head_outputs = []
                
                for t in range(T):
                    q_t = q[b, t, h]  # [D]
                    h_kk_t = h_kk_all[t]  # [D, D]
                    h_kv_t = h_kv_all[t]  # [D, D]
                    
                    # Solve: (h_kk + lambda*I) @ q_star = q using Conjugate Gradient
                    q_star = self._conjugate_gradient_solve(
                        A=h_kk_t,
                        b=q_t,
                        lamb=lamb_h,
                        max_iter=self.max_cg_iteration
                    )
                    
                    # Output: o = h_kv @ q_star
                    o_t = h_kv_t @ q_star
                    head_outputs.append(o_t)
                
                # Stack outputs for this head: [T, D]
                head_outputs = torch.stack(head_outputs, dim=0)
                batch_outputs.append(head_outputs)
            
            # Stack outputs for this batch: [T, H, D]
            batch_outputs = torch.stack(batch_outputs, dim=1)
            outputs.append(batch_outputs)
        
        # Stack all batches: [B, T, H, D]
        outputs = torch.stack(outputs, dim=0)
        
        return outputs
    
    def _conjugate_gradient_solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        lamb: torch.Tensor,
        max_iter: int = 30
    ) -> torch.Tensor:
        """
        Solve (A + lambda*I) @ x = b using Conjugate Gradient method.
        
        This is the "test-time training" aspect: iteratively refine the solution
        to adapt to the current input.
        
        Args:
            A: [D, D] - matrix
            b: [D] - right-hand side
            lamb: [D] - diagonal regularization (per dimension)
            max_iter: Maximum CG iterations
            
        Returns:
            x: [D] - solution
        """
        D = b.shape[0]
        
        # Matrix-vector product function: (A + lambda*I) @ x
        def matvec(x):
            return A @ x + lamb * x
        
        # Initial guess: x = b / (diag(A) + lambda)
        diag_A = torch.diagonal(A)
        x = b / (diag_A + lamb + 1e-8)
        
        # Initial residual: r = b - (A + lambda*I) @ x
        r = b - matvec(x)
        p = r.clone()
        delta_old = torch.dot(r, r)
        
        # CG iterations
        for i in range(max_iter):
            # Check convergence
            if delta_old < 1e-10:
                break
            
            # Compute: q = (A + lambda*I) @ p
            q = matvec(p)
            
            # Compute step size: alpha = delta_old / (p @ q)
            p_dot_q = torch.dot(p, q)
            if abs(p_dot_q) < 1e-10:
                break
            alpha = delta_old / p_dot_q
            
            # Update solution: x = x + alpha * p
            x = x + alpha * p
            
            # Update residual: r = r - alpha * q
            r = r - alpha * q
            
            # Compute new delta: delta_new = r @ r
            delta_new = torch.dot(r, r)
            
            # Compute beta: beta = delta_new / delta_old
            if delta_old < 1e-10:
                break
            beta = delta_new / delta_old
            
            # Update search direction: p = r + beta * p
            p = r + beta * p
            
            delta_old = delta_new
        
        return x
    
    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm implementation."""
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x.float() / rms * weight).to(x.dtype)
    
    def _gated_rms_norm(self, x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Gated RMSNorm: RMSNorm(x) * sigmoid(g)."""
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        x_norm = (x.float() / rms) * weight
        return (x_norm * torch.sigmoid(g.float())).to(x.dtype)


# Problem dimensions
batch_size = 4
seq_len = 512
hidden_size = 2048
num_heads = 16
head_dim = 128
use_output_gate = False
lambda_lower_bound = 0.25
max_cg_iteration = 30


def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]


def get_init_inputs():
    return [hidden_size, num_heads, head_dim, use_output_gate, lambda_lower_bound, max_cg_iteration]

