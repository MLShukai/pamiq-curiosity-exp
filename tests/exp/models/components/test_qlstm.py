import pytest
import torch

from exp.models.components.qlstm import QLSTM, LastHiddenQLSTM, scan

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

    def test_qlstm(self, qlstm):
        """Test QLSTM with provided hidden state."""
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (BATCH, DEPTH, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden_out = qlstm(x, hidden)
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, LEN, DIM)

        x, hidden_out = qlstm(x, hidden_out[:, :, -1, :])
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, LEN, DIM)

    def test_qlstm_no_hidden(self, qlstm):
        """Test QLSTM without providing hidden state."""
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)

        x, hidden_out = qlstm(x)
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, LEN, DIM)

        # Test continuing with the output hidden state
        x, hidden_out = qlstm(x, hidden_out[:, :, -1, :])
        assert x.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, LEN, DIM)


class TestLastHiddenQLSTM:
    @pytest.fixture
    def last_hidden_qlstm(self):
        last_hidden_qlstm = LastHiddenQLSTM(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT)
        return last_hidden_qlstm

    def test_last_hidden_qlstm(self, last_hidden_qlstm):
        """Test LastHiddenQLSTM with provided hidden state."""
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_element_shape = (BATCH, DIM)
        hidden = [torch.randn(*hidden_element_shape) for _ in range(DEPTH)]

        x, hidden_out = last_hidden_qlstm(x, hidden)
        assert x.shape == x_shape
        for h in hidden_out:
            assert h.shape == (BATCH, DIM)

        x, hidden_out = last_hidden_qlstm(x, hidden_out)
        assert x.shape == x_shape
        for h in hidden_out:
            assert h.shape == (BATCH, DIM)
