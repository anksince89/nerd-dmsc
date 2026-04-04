from SirenLayer import SirenLayer
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────
# SIREN MLP
# Input:  [N, 3202]  (3200 local encoding + 2 coords)
# Output: [N, 3]     (RGB per pixel)
#
# Architecture (paper Fig. 2):
#   x → Layer1 → Layer2 → [cat x] → Layer3 →
#       Layer4 → [cat x] → Layer5 → Linear → RGB
#
# Skip connections at entry of Layer2 and Layer4
# ─────────────────────────────────────────────────────────
class SirenMLP(nn.Module):
    def __init__(self,
                 in_dim:  int   = 3202,   # 3200 + 2
                 hidden:  int   = 256,
                 out_dim: int   = 3,      # RGB
                 omega_0: float = 30.0):
        super().__init__()

        self.in_dim = in_dim

        # Layer 1 — first layer, ω₀=30
        self.layer1 = SirenLayer(in_dim, hidden, omega_0=omega_0, is_first=True)

        # Layer 2 — hidden(256)
        self.layer2 = SirenLayer(hidden, hidden, omega_0=1.0)

        # Layer 3 — skip: input concatenated before this layer
        # in_dim = hidden(256) + original_input(3202) = 3458
        self.layer3 = SirenLayer(hidden + in_dim, hidden, omega_0=1.0)

        # Layer 4 — hidden(256)
        self.layer4 = SirenLayer(hidden, hidden, omega_0=1.0)

        # Layer 5 — skip: input concatenated before this layer
        self.layer5 = SirenLayer(hidden + in_dim,  hidden, omega_0=1.0)

        # Output layer — linear, no sine activation
        self.out = SirenLayer(hidden, out_dim, omega_0=1.0, is_last=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x [N, 3202]  where N = B*H*W
        Output:   [N, 3]     RGB values in [0,1]
        """
        # saving input for skip connection
        skip = x                              # [N, 3202]

        h = self.layer1(x)                    # [N, 256]
        h = self.layer2(h)                    # [N, 256]

        h = torch.cat([h, skip], dim=-1)
        h = self.layer3(h)
        # [N, 256+3202] → [N, 256]  ← skip at layer 3

        h = self.layer4(h) # [N, 256]

        h = torch.cat([h, skip], dim=-1)
        h = self.layer5(h)
        # [N, 256+3202] → [N, 256]  ← skip at layer 5

        rgb = self.out(h) # [N, 3]

        return rgb # [N, 3] in [0, 1]
    