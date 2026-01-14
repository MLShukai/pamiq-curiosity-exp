import pytest
import torch

from exp.models.components.qlstm import QLSTM
from exp.models.components.ttt import TTT

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1
NUM_HEAD = 4


class TestStackedHiddenState:
    @pytest.fixture
    def qlstm(self):
        return QLSTM(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT)

    @pytest.mark.parametrize(
        "x_shape,hidden_shape,expected_hidden_shape",
        [
            ((BATCH, LEN, DIM), (BATCH, DEPTH, DIM), (BATCH, DEPTH, LEN, DIM)),
            ((LEN, DIM), (DEPTH, DIM), (DEPTH, LEN, DIM)),
            (
                (1, 2, 3, BATCH, LEN, DIM),
                (1, 2, 3, BATCH, DEPTH, DIM),
                (1, 2, 3, BATCH, DEPTH, LEN, DIM),
            ),
        ],
    )
    def test_forward_with_hidden(
        self, qlstm, x_shape, hidden_shape, expected_hidden_shape
    ):
        """Test forward pass with provided hidden state."""
        x = torch.randn(*x_shape)
        hidden = torch.randn(*hidden_shape)

        x_out, hidden_out = qlstm(x, hidden)
        assert x_out.shape == x_shape
        assert hidden_out.shape == expected_hidden_shape

    @pytest.mark.parametrize(
        "x_shape,expected_hidden_shape",
        [
            ((BATCH, LEN, DIM), (BATCH, DEPTH, LEN, DIM)),
            ((LEN, DIM), (DEPTH, LEN, DIM)),
            ((1, 2, 3, BATCH, LEN, DIM), (1, 2, 3, BATCH, DEPTH, LEN, DIM)),
        ],
    )
    def test_forward_without_hidden(self, qlstm, x_shape, expected_hidden_shape):
        """Test forward pass without hidden state (hidden=None)."""
        x = torch.randn(*x_shape)

        x_out, hidden_out = qlstm(x)
        assert x_out.shape == x_shape
        assert hidden_out.shape == expected_hidden_shape

    @pytest.mark.parametrize(
        "x_shape,hidden_shape,error_msg",
        [
            ((BATCH, LEN, DIM), (BATCH + 1, DEPTH, DIM), "Batch shape mismatch"),
            ((BATCH, LEN, DIM), (BATCH, DEPTH, DIM + 1), "Feature dim mismatch"),
            (
                (2, 3, BATCH, LEN, DIM),
                (2, 4, BATCH, DEPTH, DIM),
                "Batch shape mismatch",
            ),
        ],
    )
    def test_shape_mismatches(self, qlstm, x_shape, hidden_shape, error_msg):
        """Test error cases when shapes don't match."""
        x = torch.randn(*x_shape)
        hidden = torch.randn(*hidden_shape)

        with pytest.raises(ValueError, match=error_msg):
            qlstm(x, hidden)


class TestStackedTTT:
    @pytest.fixture
    def ttt(self):
        return TTT(
            DEPTH,
            DIM,
            DIM_FF_HIDDEN,
            NUM_HEAD,
            base_lr=0.001,
            base_weight_decay=0.01,
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
        assert surprisal.shape == (LEN, DEPTH, NUM_HEAD)

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
