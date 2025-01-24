import torch
from lnn_zero import GoModel
from gameplay import play_against_model

if __name__ == "__main__":
    # Load the trained model
    model = GoModel()
    model.load_state_dict(torch.load("trained_go_model.pt"))
    model.eval()

    # Play the game
    play_against_model(model)
