import pytest
import torch

from exp.models.components.qlstm import QLSTM

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1


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
        "x_shape,hidden_shape,use_hidden",
        [
            ((BATCH, DIM), (BATCH, DEPTH, DIM), True),
            ((BATCH, DIM), None, False),
        ],
    )
    def test_forward_with_no_len(self, qlstm, x_shape, hidden_shape, use_hidden):
        """Test forward_with_no_len with and without hidden state."""
        x = torch.randn(*x_shape)
        hidden = torch.randn(*hidden_shape) if use_hidden else None

        x_out, hidden_out = qlstm.forward_with_no_len(x, hidden)
        assert x_out.shape == x_shape
        assert hidden_out.shape == (BATCH, DEPTH, DIM)

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
