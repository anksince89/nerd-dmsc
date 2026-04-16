import math
import torch
import torch.nn as nn

# SIREN is an MLP architecture that uses sinusoidal activations (sin(ω₀·x)) for implicit neural representations
# Key ideas: each layer applies a sine nonlinearity (scaled by ω₀) instead of ReLU, 
# enabling modeling of high-frequency details. 
# The hyperparameter ω₀ (omega_0) controls the input frequency scale; 
# typically ω₀≈30 for the first layer and 1–5 for subsequent layers. 
# A special initialization (weights drawn from uniform[-1/in,1/in] for first layer,
#  and [-√(6/in)/ω₀, +] for hidden layers) preserves activation distributions, ensuring stable training.
# Reference paper: https://arxiv.org/abs/2006.09661
class SirenLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 omega_0: float = 1.0, 
                 is_first: bool = False,
                 is_last: bool = False):
        super().__init__()
        self.omega_0    = omega_0
        self.is_first   = is_first
        self.is_last     = is_last
        self.in_features = in_dim
        self.linear     = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            elif self.is_last:
                bound = math.sqrt(6.0 / self.in_features)
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            nn.init.uniform_(self.linear.weight, -bound, bound)
            if self.linear.bias is not None:
                nn.init.uniform_(self.linear.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.is_last:
            return out

        out.mul_(self.omega_0)
        out.sin_()
        return out
