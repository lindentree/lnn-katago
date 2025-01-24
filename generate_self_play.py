import torch
from mcts import MCTSAgent
from lnn_zero import GoModel

# Parameters
BOARD_SIZE = 19
NUM_GAMES = 10  # Adjust the number of self-play games as needed
batch_size = 16
history_length = 8
FILENAME = 'self_play_data.h5'

def main():
    # Initialize model and MCTS
    model = GoModel(board_size=BOARD_SIZE, history_length=history_length)
    # Create an initial empty board state (adjust according to your game rules)
    state = torch.rand((batch_size, history_length, BOARD_SIZE, BOARD_SIZE))  # Random board state
    input_tensor = torch.rand((batch_size, history_length, BOARD_SIZE, BOARD_SIZE))
    valid_moves_mask = torch.ones((batch_size, BOARD_SIZE * BOARD_SIZE), dtype=torch.bool)  # Batch size of 1, single-channel board

    policy_logits, value = model(input_tensor, valid_moves_mask)
    print("Policy Logits Shape:", policy_logits.shape)  # [batch_size, board_size**2]
    print("Value Shape:", value.shape)  # [batch_size, 1]

    # Generate self-play data
    agent = MCTSAgent(model, num_rounds=800, temperature=1.0)
    agent.generate_self_play_data(agent, NUM_GAMES, BOARD_SIZE, 'self_play_data.h5')
    print(f"Self-play data saved to {FILENAME}")

if __name__ == "__main__":
    main()
