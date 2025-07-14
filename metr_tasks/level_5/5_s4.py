# from https://github.com/HazyResearch/safari/blob/main/src/models/sequence/ssm/s4_simple.py
import torch
import torch.nn as nn
from einops import rearrange, repeat
import opt_einsum as oe

import math
class OurModule(nn.Module):
    def __init__(self): super().__init__()

    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None: optim["lr"] = lr
        if trainable and wd is not None: optim["weight_decay"] = wd
        if len(optim) > 0: setattr(getattr(self, name), "_optim", optim)

#
# This is intended to match np.convolve(x,w)[:len(w)]
# That is, (u \ast v)[k] = sum_{j} u[k-j]v[j]
# Here y = (u \ask v) on return.
# We assume the inputs are:
# u (B H L)
# v (C H L)
# and we want to produce y that is (B C H L)
#


def fft_conv(u,v):
    L   = u.shape[-1]
    u_f = torch.fft.rfft(u, n=2*L) # (B H L)
    v_f = torch.fft.rfft(v, n=2*L) # (C H L)
   
    y_f = oe.contract('bhl,chl->bchl', u_f, v_f) 
    y   = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)
    return y

def normalize_param(a, method, norm_const=None):
        if method == "l1":
            if norm_const is not None:
                return a/((1+norm_const)*torch.linalg.norm(a,ord=1,dim=2).unsqueeze(2))
            return a/torch.linalg.norm(a,ord=1,dim=2).unsqueeze(2)
        if method == "l2":
            return a/torch.linalg.norm(a,ord=2,dim=2).unsqueeze(2)
        if method == "max":
            return 0.1*a/torch.max(a,dim=2)[0].unsqueeze(2)
        if method == "none":
            return a
        raise ValueError(f"{method} normalization not implemented")

class Model(OurModule):
    def __init__(self,
            nHippos,
            d_state=64,
            channels=1, 
            use_initial=True, # Use the initial state?
            zero_order_hold=False, # Use zero-order hold approximation
            trap_rule=True,
            dt_min=0.001,
            dt_max=0.1,
            lr=None, # Hook to set LR of SSM parameters differently
            learn_a=True,
            learn_theta=True,
            learn_dt=False, # whether to learn separate dt for each hippo
            theta_scale=False,
            skip_connection=True,
            repr='cont', # representation to use: ['cont','disc','comp'] 
            param_norm = 'none', # for normalizing parameters for stability
            **kernel_args,): # Use the trapezoid rule
        super().__init__()
        # H is number of hippos
        # D is the dimension (also shockingly n other places)
        # B is the batch
        # L is the length
        self.h = nHippos
        self.d = d_state // 2    
        self.channels = channels
        self.use_initial = use_initial
        self.zero_order_hold = zero_order_hold
        #
        # Use the trapezoid rule correct or just do zero-order hold.
        self.trap_rule = trap_rule
        self.repr = repr
        self.learn_dt = learn_dt
        self.shift = 'shift' in self.repr
        self.param_norm = param_norm

        _fp    = (self.channels, self.h, self.d)
        
        # Chebyshev initialization
        h_scale  = torch.exp(torch.arange(self.h)/self.h * math.log(dt_max/dt_min))
        angles   = torch.arange(self.d)*torch.pi
        t_scale  = h_scale if theta_scale else torch.ones(self.h)
        theta    = oe.contract('c,h,d->chd', torch.ones(self.channels), t_scale, angles)
        if self.repr == 'disc':
            # discrete diagonal representation
            a = torch.randn(*_fp).abs()
            #a = 2*torch.rand(*_fp)-1 # init randomly from [-1,1]
        else:
            # default continuous diagonal representation
            a = -repeat(h_scale, 'h -> c h d', c=self.channels, d=self.d)
                                            
        self.register("theta", theta,learn_theta,lr=lr, wd=None)
        self.register("a", a, learn_a,lr=lr, wd=None)

        if self.learn_dt:
            log_dt = torch.rand(self.h) * (
                math.log(dt_max) - math.log(dt_min)
            ) + math.log(dt_min)
            self.register("log_dt", log_dt, True,lr=lr, wd=None)

        # The other maps 
        if not skip_connection:
            self.register("D", torch.zeros((channels, self.h)), False)
        else:
            self.D = nn.Parameter(torch.randn(channels, self.h))
        
        if use_initial or 'comp' in self.repr:
            if self.shift:
                b = torch.zeros(*_fp)
                b[:,:,0] = 1
                self.register("b", b, False)
            else:
                self.b = nn.Parameter(torch.randn(*_fp))
            self.c = nn.Parameter(torch.randn(*_fp))
            self.x0 = nn.Parameter(torch.randn(*_fp))
        else:
            # This is an optimization that we combine q = c * b
            # It's as if we're setting x0 = 0.
            self.q = nn.Parameter(torch.randn(*_fp))


    def quadrature_method(self, u, horizon):
        # The input is now Batch x Hippos x Length
        l  = u.size(-1)

        dt = 1/(l-1) # the step size
        if self.learn_dt:
            dt = torch.exp(self.log_dt).view(1,-1,1, 1)

        # q and a are both C x H x D
        # zk is of length l we want a C x H x L matrix
        zk = dt*torch.arange(l, device=u.device).view(1,1,-1,1)

        if self.repr == 'disc':
            # discrete diagonal representation
            a_ = (self.a).abs()
            base_term = 2 * dt * torch.pow(a_.unsqueeze(2), zk) * torch.cos(self.theta.unsqueeze(2) * zk)
        else:
            # continuous diagonal representation
            a_ = self.a #/torch.linalg.norm(self.a,ord=1,dim=2).unsqueeze(2)
            a_ = -a_.abs()
            # a_ = -self.a.abs()
            base_term = 2*dt*torch.exp(a_.unsqueeze(2) * zk)*torch.cos(   self.theta.unsqueeze(2) * zk)

        q  = self.b*self.c if self.use_initial else self.q
        f  = (q.unsqueeze(2)*base_term).sum(-1)

        y = fft_conv(u,f)
        # Add in the skip connection with per-channel D matrix
        y = y + oe.contract('bhl,ch->bchl', u, self.D)
        # Add back the initial state
        if self.use_initial:
            y = y + (2*(self.c*self.x0).unsqueeze(2)*base_term).sum(-1)

        return rearrange(y, 'b c h l-> b (c h) l') # flatten the channels.

    def forward(self, u, horizon=None):
        return self.quadrature_method(u, horizon)

    
batch_size = 8
nHippos = 16
seq_len = 256

def get_inputs():
    return [torch.randn(batch_size, nHippos, seq_len)]

def get_init_inputs():
    return [nHippos]

if __name__ == "__main__":
    torch.set_default_device("cuda")
    model = Model(nHippos)
    inputs = get_inputs()
    print(model(*inputs).shape)