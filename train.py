# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import DataLoader, Dataset
from lnn_zero import GoModel
from mcts import MCTS

# Parameters
BOARD_SIZE = 19
LEARNING_RATE = 1e-3
EPOCHS = 10
SIMULATIONS = 800

class GoDataset(Dataset):
    def __init__(self, h5_file):
        self.data = []
        with h5py.File(h5_file, 'r') as f:
            for game in f.values():
                sequence = torch.tensor(game['sequence'][:], dtype=torch.float32)
                policy = torch.tensor(game['policy'][:], dtype=torch.float32)
                value = torch.tensor(game['value'][()], dtype=torch.float32)
                self.data.append((sequence, policy, value))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



# Training function
def train(model, data_loader, optimizer, loss_fn, device):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for sequences, target_policy, target_value in data_loader:
            # Move data to the device
            sequences = sequences.to(device)  # Shape: (batch_size, seq_len, board_size, board_size)
            target_policy = target_policy.to(device)  # Shape: (batch_size, board_size**2)
            target_value = target_value.to(device)  # Shape: (batch_size, 1)

            optimizer.zero_grad()

            # Flatten policy_logits for the output of the LSTM model
            policy_logits, value = model(sequences)  # Shape: (batch_size, board_size**2), (batch_size, 1)

            # Calculate losses
            loss_policy = loss_fn(policy_logits, target_policy)
            loss_value = F.mse_loss(value, target_value)
            loss = loss_policy + loss_value

            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoModel(board_size=BOARD_SIZE).to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # Replace with actual self-play data
    dataset = GoDataset('self_play_data.h5')
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    train(model, data_loader, optimizer, loss_fn, device)