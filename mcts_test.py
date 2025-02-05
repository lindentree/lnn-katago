import pytest
from mcts import MCTSNode, MCTSAgent
from game import GameState  # Assuming you have a GameState class managing the Go board.
import torch
import h5py

class MockModel:
    def __call__(self, state_tensor, valid_moves_mask=None):
        batch_size, board_size = state_tensor.shape[0], state_tensor.shape[-1]
        logits = torch.zeros((batch_size, board_size * board_size))
        value = torch.tensor([0.5] * batch_size)
        return logits, value

@pytest.fixture
def initial_state():
    return GameState.new_game(3)  # Assuming a 3x3 board for simplicity.

@pytest.fixture
def model():
    return MockModel()

@pytest.fixture
def agent(model):
    return MCTSAgent(model, num_rounds=10, temperature=1.0)

def test_mcts_node_initialization(initial_state):
    node = MCTSNode(initial_state)
    assert node.game_state == initial_state
    assert node.parent is None
    assert node.move is None
    assert node.win_counts == {1: 0, -1: 0}
    assert node.num_rollouts == 0
    assert node.children == []
    assert node.unvisited_moves == initial_state.legal_moves()

def test_mcts_node_add_child(initial_state):
    node = MCTSNode(initial_state)
    move = initial_state.legal_moves()[0]
    new_state = initial_state.apply_move(move)
    child = node.add_child(move, new_state)
    assert child.parent == node
    assert child.move == move
    assert child.game_state == new_state
    assert move not in node.unvisited_moves
    assert child in node.children

def test_mcts_node_record_win(initial_state):
    node = MCTSNode(initial_state)
    node.record_win(1)
    assert node.win_counts[1] == 1
    assert node.num_rollouts == 1

def test_mcts_node_can_add_child(initial_state):
    node = MCTSNode(initial_state)
    assert node.can_add_child() == (len(initial_state.legal_moves()) > 0)

def test_mcts_node_is_terminal(initial_state):
    node = MCTSNode(initial_state)
    assert node.is_terminal() == initial_state.is_over()

def test_mcts_node_winning_pct(initial_state):
    node = MCTSNode(initial_state)
    assert node.winning_pct(1) == 0
    node.record_win(1)
    assert node.winning_pct(1) == 1.0

def test_mcts_agent_select_move(agent, initial_state):
    move = agent.select_move(initial_state)
    assert move in initial_state.legal_moves()

def test_mcts_agent_simulate(agent, initial_state):
    node = MCTSNode(initial_state)
    reward = agent.simulate(node)
    assert reward == 0.5

def test_mcts_agent_generate_self_play_data(agent):
    filename = 'self_play_data.h5'
    agent.generate_self_play_data(agent, num_games=1, board_size=3, filename=filename)
    with h5py.File(filename, 'r') as f:
        assert 'states' in f
        assert 'policies' in f
        assert 'values' in f
