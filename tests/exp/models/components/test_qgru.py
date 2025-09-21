import pytest
import torch

from exp.models.components import qlstm
from exp.models.components.qgru import QGRU

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1


class TestQGRU:
    @pytest.fixture
    def qgru(self):
        qgru = QGRU(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT)
        return qgru

    def test_qgru(self, qgru):
        """Test QGRU with provided hidden state."""
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (BATCH, DEPTH, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden_out = qgru(x, hidden)
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, LEN, DIM)

        x, hidden_out = qgru(x, hidden_out[:, :, -1, :])
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, LEN, DIM)

    def test_qgru_no_hidden(self, qgru):
        """Test QGRU without providing hidden state."""
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)

        x, hidden_out = qgru(x)
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, LEN, DIM)

        # Test continuing with the output hidden state
        x, hidden_out = qgru(x, hidden_out[:, :, -1, :])
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, LEN, DIM)
