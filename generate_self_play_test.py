import os
import h5py
import torch
import pytest
from generate_self_play import GoModel, MCTSAgent, BOARD_SIZE, NUM_GAMES, FILENAME

@pytest.fixture
def setup():
    """Fixture to set up the model, agent, and test inputs."""
    model = GoModel(board_size=BOARD_SIZE, history_length=8)
    agent = MCTSAgent(model, num_rounds=800, temperature=1.0)
    batch_size = 16
    history_length = 8
    state = torch.rand((batch_size, history_length, BOARD_SIZE, BOARD_SIZE))
    input_tensor = torch.rand((batch_size, history_length, BOARD_SIZE, BOARD_SIZE))
    valid_moves_mask = torch.ones((batch_size, BOARD_SIZE * BOARD_SIZE), dtype=torch.bool)
    yield model, agent, batch_size, history_length, state, input_tensor, valid_moves_mask

    # Cleanup: Remove generated file
    if os.path.exists(FILENAME):
        try:
            os.remove(FILENAME)
        except OSError as e:
            print(f"Error removing {FILENAME}: {e}")

def test_model_output_shapes(setup):
    """Test that the model produces outputs of the correct shapes."""
    model, agent, batch_size, history_length, state, input_tensor, valid_moves_mask = setup
    policy_logits, value = model(input_tensor, valid_moves_mask)

    # Check policy_logits shape
    assert policy_logits.shape == (batch_size, BOARD_SIZE * BOARD_SIZE), \
        f"Expected policy_logits shape {(batch_size, BOARD_SIZE * BOARD_SIZE)}, got {policy_logits.shape}"

    # Check value shape
    assert value.shape == (batch_size, 1), \
        f"Expected value shape {(batch_size, 1)}, got {value.shape}"

def test_generate_self_play_data(setup):
    """Test that self-play data is generated and saved correctly."""
    model, agent, batch_size, history_length, state, input_tensor, valid_moves_mask = setup
    agent.generate_self_play_data(agent, NUM_GAMES, BOARD_SIZE, FILENAME)

    # Check if file exists
    assert os.path.exists(FILENAME), f"File {FILENAME} was not created"

    # Validate file content
    with h5py.File(FILENAME, 'r') as f:
        assert 'states' in f, "'states' dataset missing in HDF5 file"
        assert 'policies' in f, "'policies' dataset missing in HDF5 file"
        assert 'values' in f, "'values' dataset missing in HDF5 file"

        states = f['states']
        policies = f['policies']
        values = f['values']

        # Validate dataset shapes
        assert states.shape[1:] == (history_length, BOARD_SIZE, BOARD_SIZE), \
            f"Expected states shape [history_length, {BOARD_SIZE}, {BOARD_SIZE}], got {states.shape[1:]}"
        assert policies.shape[1:] == (BOARD_SIZE * BOARD_SIZE,), \
            f"Expected policies shape [{BOARD_SIZE * BOARD_SIZE}], got {policies.shape[1:]}"
        assert values.shape[1:] == (1,), \
            f"Expected values shape [1], got {values.shape[1:]}"

def test_edge_cases():
    """Test edge cases like small boards and invalid inputs."""
    model = GoModel(board_size=9, history_length=4)  # Small board, short history
    input_tensor = torch.rand((1, 4, 9, 9))  # Batch size 1, small history
    valid_moves_mask = torch.ones((1, 81), dtype=torch.bool)

    # Check that model runs without errors
    policy_logits, value = model(input_tensor, valid_moves_mask)
    assert policy_logits.shape == (1, 81), f"Unexpected policy_logits shape {policy_logits.shape}"
    assert value.shape == (1, 1), f"Unexpected value shape {value.shape}"
