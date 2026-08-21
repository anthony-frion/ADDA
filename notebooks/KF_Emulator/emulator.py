import neuralop
import torch
from torch import nn


class Emulator(nn.Module):
    """Emulator class that loads a pretrained FNO2d model and provides a method for integrating the state forward in
    time."""

    def __init__(self, pretrained_checkpoint_path=None):
        super().__init__()
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.network = neuralop.models.FNO2d(
            n_modes_height=16,
            n_modes_width=16,
            hidden_channels=128,
            in_channels=2,
            out_channels=2,
            lifting_channels=128,
            projection_channels=128,
            n_layers=4,
            non_linearity=nn.SiLU(),
        )
        checkpoint = torch.load(self.pretrained_checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
        self.network.load_state_dict(checkpoint, strict=True)
        for param in self.network.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Forward pass through the emulator network."""
        return self.network(x)

    def integrate(self, state, time):
        """Integrate the state forward in time using the emulator."""
        rollout = []
        for _ in range(len(time)):
            # print(state.shape)  # should be (B, 2, 32, 32)
            state = self.forward(state)
            rollout.append(state)
        rollout = torch.stack(rollout, dim=1)
        return rollout
