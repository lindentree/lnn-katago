import pytest
import torch
from lnn_zero import GoModel

class TestGoModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.board_size = 19
        self.history_length = 8
        self.model = GoModel(board_size=self.board_size, history_length=self.history_length)

    def test_input_shape(self):
        batch_size = 4
        x = torch.randn(batch_size, self.history_length, self.board_size, self.board_size)
        policy_logits, value = self.model(x)
        
        # Check policy logits shape
        assert policy_logits.shape == (batch_size, self.board_size * self.board_size)
        
        # Check value shape
        assert value.shape == (batch_size, 1)

    def test_invalid_input_shape(self):
        batch_size = 4
        invalid_x = torch.randn(batch_size, self.history_length + 1, self.board_size, self.board_size)
        
        with pytest.raises(AssertionError):
            self.model(invalid_x)

    def test_valid_moves_mask(self):
        batch_size = 4
        x = torch.randn(batch_size, self.history_length, self.board_size, self.board_size)
        valid_moves_mask = torch.ones(batch_size, self.board_size * self.board_size, dtype=torch.bool)
        valid_moves_mask[:, 0] = 0  # Mask out the first move
        
        policy_logits, _ = self.model(x, valid_moves_mask=valid_moves_mask)
        
        # Check if the first move is masked
        assert (policy_logits[:, 0] == -1e9).all()

if __name__ == '__main__':
    unittest.main()