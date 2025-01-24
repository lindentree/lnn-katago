import torch
import torch.nn as nn
import torch.nn.functional as F

class LiquidLayer(nn.Module):
    def __init__(self, input_size, num_units):
        super(LiquidLayer, self).__init__()
        self.num_units = num_units

        # Trainable parameters for time constants and weights
        self.time_constants = nn.Parameter(torch.randn(num_units))
        self.input_weights = nn.Linear(input_size, num_units)
        self.recurrent_weights = nn.Linear(num_units, num_units)
        self.bias = nn.Parameter(torch.zeros(num_units))

    def forward(self, x):
        if x.dim() == 2:
            raise ValueError(
                f"Expected 3D input [batch_size, time_steps, features], but got {x.shape}"
            )
        # Initialize hidden state
        h = torch.zeros(x.size(0), self.num_units, device=x.device)

        # Iterate over the sequence
        for t in range(x.size(1)):
            input_t = x[:, t, :]  # Extract the t-th timestep
            h = h + self.time_constants * torch.tanh(
                self.input_weights(input_t) + self.recurrent_weights(h) + self.bias
            )
        return h


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class GoModel(nn.Module):
    def __init__(self, board_size, history_length):
        super(GoModel, self).__init__()
        self.board_size = board_size
        self.history_length = history_length

        # Convolutional layers for spatial-temporal feature extraction
        self.conv = nn.Conv3d(
            in_channels=history_length,  # Temporal channels
            out_channels=16,
            kernel_size=(1, 3, 3),  # Spatial kernel
            padding=(0, 1, 1)
        )

        self.mish = Mish()
        self.flattened_feature_size = 16 * board_size * board_size  # Calculate feature size
        self.liquid_layer = LiquidLayer(
            input_size=self.flattened_feature_size,
            num_units=128
        )

        # Heads for policy and value outputs
        self.policy_head = nn.Linear(128, board_size * board_size)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, valid_moves_mask=None):
        # Input shape: [batch_size, history_length, board_size, board_size]
        print(f"Input shape: {x.shape}, Expected history_length: {self.history_length}")

        assert x.size(1) == self.history_length, "Input channels must match history_length"

        # Add a depth dimension for Conv3d: [batch_size, channels, depth, height, width]
        x = x.unsqueeze(2)
        x = self.mish(self.conv(x))  # [batch_size, 16, history_length, board_size, board_size]

        # Flatten spatial dimensions but preserve batch and time: [batch_size, history_length, features]
        batch_size, _, history_length, _, _ = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, history_length, 16, board_size, board_size]
        x = x.view(batch_size, history_length, -1)  # Flatten: [batch_size, history_length, features]

        # Pass through the liquid layer: input shape [batch_size, time_steps, features]
        x = self.liquid_layer(x)

        # Policy head
        policy_logits = self.policy_head(x)  # [batch_size, board_size * board_size]

        # Mask invalid moves in policy logits
        if valid_moves_mask is not None:
            policy_logits = policy_logits.masked_fill(~valid_moves_mask, -1e9)

        # Value head
        value = torch.tanh(self.value_head(x))  # [batch_size, 1]

        return policy_logits, value






