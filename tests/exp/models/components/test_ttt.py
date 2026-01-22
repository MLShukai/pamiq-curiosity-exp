import pytest
import torch

from exp.models.components.ttt import TTT

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1
NUM_HEAD = 4


class TestTTT:
    @pytest.fixture
    def ttt(self):
        return TTT(
            DEPTH,
            DIM,
            DIM_FF_HIDDEN,
            NUM_HEAD,
            base_lr=(0.0001, 0.01),
            chunk_size=16,
            dropout=DROPOUT,
        )

    def test_forward_with_hidden(self, ttt):
        """Test forward pass with provided hidden state with batch."""
        x = torch.randn(BATCH, LEN, DIM)
        hidden = [
            {
                "W1": torch.randn(
                    BATCH, NUM_HEAD, DIM_FF_HIDDEN // NUM_HEAD, DIM // NUM_HEAD
                ),
                "W2": torch.randn(
                    BATCH, NUM_HEAD, DIM // NUM_HEAD, DIM_FF_HIDDEN // NUM_HEAD
                ),
            }
            for _ in range(DEPTH)
        ]

        x_out, hidden_out, surprisal = ttt(x, hidden)
        assert x_out.shape == x.shape
        assert len(hidden_out) == DEPTH
        assert all(
            hidden_out[i]["W1"].shape == hidden[i]["W1"].shape
            and hidden_out[i]["W2"].shape == hidden[i]["W2"].shape
            for i in range(DEPTH)
        )
        assert surprisal.shape == (BATCH, LEN, DEPTH, NUM_HEAD)

    def test_forward_with_no_hidden(self, ttt):
        """Test forward pass without hidden state, but with batch."""
        x = torch.randn(BATCH, LEN, DIM)

        x_out, hidden_out, surprisal = ttt(x)
        assert x_out.shape == x.shape
        assert len(hidden_out) == DEPTH
        assert all(
            hidden_out[i]["W1"].shape
            == (BATCH, NUM_HEAD, DIM_FF_HIDDEN // NUM_HEAD, DIM // NUM_HEAD)
            and hidden_out[i]["W2"].shape
            == (BATCH, NUM_HEAD, DIM // NUM_HEAD, DIM_FF_HIDDEN // NUM_HEAD)
            for i in range(DEPTH)
        )
        assert surprisal.shape == (BATCH, LEN, DEPTH, NUM_HEAD)

    def test_forward_with_hidden_no_batch(self, ttt):
        """Test forward pass with provided hidden state without batch."""
        x = torch.randn(LEN, DIM)
        hidden = [
            {
                "W1": torch.randn(NUM_HEAD, DIM_FF_HIDDEN // NUM_HEAD, DIM // NUM_HEAD),
                "W2": torch.randn(NUM_HEAD, DIM // NUM_HEAD, DIM_FF_HIDDEN // NUM_HEAD),
            }
            for _ in range(DEPTH)
        ]

        x_out, hidden_out, surprisal = ttt(x, hidden)
        assert x_out.shape == x.shape
        assert len(hidden_out) == DEPTH
        assert all(
            hidden_out[i]["W1"].shape == hidden[i]["W1"].shape
            and hidden_out[i]["W2"].shape == hidden[i]["W2"].shape
            for i in range(DEPTH)
        )

    def test_forward_with_no_hidden_no_batch(self, ttt):
        """Test forward pass without hidden state and without batch."""
        x = torch.randn(LEN, DIM)

        x_out, hidden_out, surprisal = ttt(x)
        assert x_out.shape == x.shape
        assert len(hidden_out) == DEPTH
        assert all(
            hidden_out[i]["W1"].shape
            == (NUM_HEAD, DIM_FF_HIDDEN // NUM_HEAD, DIM // NUM_HEAD)
            and hidden_out[i]["W2"].shape
            == (NUM_HEAD, DIM // NUM_HEAD, DIM_FF_HIDDEN // NUM_HEAD)
            for i in range(DEPTH)
        )
        assert surprisal.shape == (LEN, DEPTH, NUM_HEAD)
