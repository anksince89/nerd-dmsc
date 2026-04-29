from SirenLayer import SirenLayer
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────
# SIREN MLP
# Input:  [N, C * patch_size^2 + 2]
# Output: [N, out_dim]     (per-pixel prediction)
#
# Architecture (paper Fig. 2):
#   x → Layer1 → Layer2 → [cat x] → Layer3 →
#       Layer4 → [cat x] → Layer5 → Linear → RGB
#
# Skip connections at entry of Layer2 and Layer4
# ─────────────────────────────────────────────────────────
class SirenMLP(nn.Module):
    def __init__(self,
                 in_dim:  int   = 3202,   # local encoding + 2
                 hidden:  int   = 256,
                 out_dim: int   = 3,      # output channels
                 omega_0: float = 30.0):
        super().__init__()

        self.in_dim = in_dim
        self.feature_dim = in_dim - 2

        # The author release uses omega_0=30 for all hidden sine layers and
        # does not mark the first layer as "is_first" for initialization.
        self.layer1 = SirenLayer(in_dim, hidden, omega_0=omega_0, is_first=False)

        # Layer 2 — hidden(256)
        self.layer2 = SirenLayer(hidden, hidden, omega_0=omega_0)

        # Layer 3 — skip: input feature vector concatenated before this layer.
        self.layer3 = SirenLayer(hidden + in_dim - 2, hidden, omega_0=omega_0)

        # Layer 4 — hidden(256)
        self.layer4 = SirenLayer(hidden, hidden, omega_0=omega_0)

        # Layer 5 — skip: input concatenated before this layer
        self.layer5 = SirenLayer(hidden + in_dim - 2,  hidden, omega_0=omega_0)

        # Output layer — author's release uses a plain Linear head.
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x [N, in_dim]  where N = B*H*W
        Output: [N, out_dim]
        """
        # Match the author implementation: only the local feature vector
        # is reused in the MLP skip connections, not the (x, y) coordinates.
        skip = x[..., :self.feature_dim]

        h = self.layer1(x)
        h = self.layer2(h)

        h = torch.cat([skip, h], dim=-1)
        h = self.layer3(h)
        # [N, feature_dim + 256] → [N, 256]  ← skip at layer 3

        h = self.layer4(h) # [N, 256]

        h = torch.cat([skip, h], dim=-1)
        h = self.layer5(h)
        # [N, feature_dim + 256] → [N, 256]  ← skip at layer 5

        out = self.out(h) # [N, out_dim]

        return out
    
