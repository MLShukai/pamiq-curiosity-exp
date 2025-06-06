import pytest
import torch

from pamiq_curiosity_exp.models.components.qlstm import QLSTM, scan

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1


class TestScan:
    def test_scan(self):
        a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        b = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        expected = torch.tensor([[1, 4, 15, 64, 325, 1956, 13699, 109600]])
        for i in range(1, 9):
            torch.allclose(scan(a[:, :i], b[:, :i]), expected[:, :i])


class TestQLSTM:
    @pytest.fixture
    def qlstm(self):
        qlstm = QLSTM(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT)
        return qlstm

    def test_qlstm_with_hidden(self, qlstm):
        """Test QLSTM with provided hidden state."""
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (BATCH, DEPTH, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden_out = qlstm(x, hidden)
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, DIM)

        x, hidden_out = qlstm(x, hidden_out)
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, DIM)

    def test_qlstm_without_hidden(self, qlstm):
        """Test QLSTM with None hidden state (auto-initialization)."""
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)

        x, hidden_out = qlstm(x, None)
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, DIM)

        # Test again with previous output
        x, hidden_out2 = qlstm(x, hidden_out)
        assert x.shape == x_shape
        assert hidden_out2.shape == (BATCH, DEPTH, DIM)
