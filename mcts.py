# mcts.py
import math
import random
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import os

from game import GameState  # Assuming you have a GameState class managing the Go board.

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {1: 0, -1: 0}  # Assuming 1 for black, -1 for white.
        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = game_state.legal_moves()  # Modify to fit your GameState class.

    def add_child(self, move, game_state):
        """Add a child node with the given move."""
        child = MCTSNode(game_state, parent=self, move=move)
        self.children.append(child)
        self.unvisited_moves.remove(move)
        return child

    def record_win(self, winner):
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    def can_add_child(self):
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        return self.game_state.is_over()

    def winning_pct(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts) if self.num_rollouts > 0 else 0

class MCTSAgent:
    def __init__(self, model, num_rounds, temperature=1.0):
        self.model = model
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state):
        root = MCTSNode(game_state)

        for _ in range(self.num_rounds):
            node = root
            while not node.can_add_child() and not node.is_terminal():
                node = self.select_child(node)

            if node.can_add_child():
                move = random.choice(node.unvisited_moves)
                new_game_state = game_state.apply_move(move)
                node = node.add_child(move, new_game_state)

            # Simulate the game from this node
            reward = 1 if self.simulate(node) > 0 else -1
            while node is not None:
                node.record_win(reward)
                node = node.parent

        best_move = max(
            root.children, key=lambda child: child.winning_pct(game_state.next_player)
        ).move
        print("Best Move:", best_move)
        return best_move

    def select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = math.log(total_rollouts)

        def uct_score(child):
            win_pct = child.winning_pct(node.game_state.next_player)
            exploration = math.sqrt(log_rollouts / (1 + child.num_rollouts))
            return win_pct + self.temperature * exploration

        return max(node.children, key=uct_score)

    def simulate(self, node):
        state = node.game_state.board  # Replace with the board representation used by your model.
        valid_moves_mask = np.ones(state.size)  # Adjust mask for valid moves.

        # Convert to tensor for model input
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        valid_moves_tensor = torch.tensor(valid_moves_mask, dtype=torch.bool).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.model(state_tensor, valid_moves_mask=valid_moves_tensor)

        probs = F.softmax(logits, dim=-1).numpy().flatten()
        return value.item()

    def generate_self_play_data(self, agent, num_games, board_size, filename):
        data = []
        for _ in range(num_games):
            game = GameState.new_game(board_size)
            game_states = []
            policies = []
            values = []

            while not game.is_over():
                # Agent selects a move
                move  = agent.select_move(game)
                policy = agent.select_move(game)

                print("Move:", move, policy)  # Debug: print the move
                

                # Store game state and policy
                game_states.append(game.board.copy())
                policies.append(policy)

                # Apply the move
                game = game.apply_move(move)

            # Determine the game result
            winner = game.winner()
            for state in game_states:
                value = 1.0 if winner == 1 else -1.0 if winner == -1 else 0.0
                values.append(value)

            # Append results for this game
            data.append((np.array(game_states), np.array(policies), np.array(values)))
            print(filename, "Game result:", winner)

        # Save to HDF5 file
        with h5py.File(filename, 'w') as f:
            print("Saving self-play data to:", filename)
            states = np.concatenate([d[0] for d in data])
            policies = np.concatenate([d[1] for d in data])
            values = np.concatenate([d[2] for d in data])

            f.create_dataset('states', data=states)
            f.create_dataset('policies', data=policies)
            f.create_dataset('values', data=values)

        if os.path.exists(filename):
            print(f"Self-play data saved to {filename}")
