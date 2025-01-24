import numpy as np

class GameState:
    def __init__(self, board_size=19):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)  # 0: empty, 1: black, -1: white
        self.next_player = 1  # 1: black, -1: white
        self.history = []  # List of past board states for undo or ko checking
        self.game_over = False

    def legal_moves(self):
        """Return a list of all legal moves as (row, col) tuples."""
        if self.game_over:
            return []
        moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] == 0:  # Empty spot
                    moves.append((row, col))
        return moves

    def apply_move(self, move):
        """
        Apply a move to the game state.
        Args:
            move (tuple): (row, col) of the move to play.
        Returns:
            GameState: A new GameState after applying the move.
        """

        print(move)
        if not move or self.game_over or self.board[move[0], move[1]] != 0:
            raise ValueError("Illegal move!")

        # Create a copy of the current state
        new_state = GameState(self.board_size)
        new_state.board = self.board.copy()
        new_state.next_player = -self.next_player  # Switch player
        new_state.history = self.history + [self.board.copy()]

        # Play the move
        row, col = int(move[0]), int(move[1])  # Ensure indices are integers
        new_state.board[row, col] = self.next_player

        # Check game-over condition (example: no more moves)
        if not new_state.legal_moves():
            new_state.game_over = True

        return new_state


    def is_over(self):
        """Check if the game is over."""
        return self.game_over

    def winner(self):
        """
        Determine the winner.
        Returns:
            int: 1 for black, -1 for white, 0 for draw, None if not over.
        """
        if not self.game_over:
            return None
        black_score = np.sum(self.board == 1)
        white_score = np.sum(self.board == -1)
        if black_score > white_score:
            return 1
        elif white_score > black_score:
            return -1
        else:
            return 0
        
    @staticmethod
    def new_game(board_size):
        return GameState(board_size)
