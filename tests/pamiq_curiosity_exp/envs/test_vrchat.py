import pytest
import torch
from pamiq_vrchat.actuators import OscAxes, OscButtons

from pamiq_curiosity_exp.envs.vrchat import OSC_ACTION_CHOICES, OscTransform


class TestOscTransform:
    """Tests for the OscTransform class."""

    def test_init(self):
        """Test initialization."""
        OscTransform()

    def test_init_with_custom_velocities(self):
        """Test initialization with custom velocity parameters."""
        transform = OscTransform(
            vertical_velocity=2.0, horizontal_velocity=1.5, look_horizontal_velocity=0.8
        )
        assert transform.vertical_velocity == 2.0
        assert transform.horizontal_velocity == 1.5
        assert transform.look_horizontal_velocity == 0.8

    @pytest.mark.parametrize("v_idx,v_val", [(0, 0.0), (1, 1.0), (2, -1.0)])
    @pytest.mark.parametrize("h_idx,h_val", [(0, 0.0), (1, 1.0), (2, -1.0)])
    @pytest.mark.parametrize("look_h_idx,look_h_val", [(0, 0.0), (1, 1.0), (2, -1.0)])
    @pytest.mark.parametrize("jump", [0, 1])
    @pytest.mark.parametrize("run", [0, 1])
    def test_call_valid_input(
        self, v_idx, v_val, h_idx, h_val, look_h_idx, look_h_val, jump, run
    ):
        """Test transformation with valid input tensor."""
        transform = OscTransform()
        action = torch.tensor([v_idx, h_idx, look_h_idx, jump, run], dtype=torch.long)

        result = transform(action)

        assert "axes" in result
        assert "buttons" in result
        assert result["axes"][OscAxes.Vertical] == v_val
        assert result["axes"][OscAxes.Horizontal] == h_val
        assert result["axes"][OscAxes.LookHorizontal] == look_h_val
        assert result["buttons"][OscButtons.Jump] is bool(jump)
        assert result["buttons"][OscButtons.Run] is bool(run)

    def test_call_with_custom_velocities(self):
        """Test transformation with custom velocity multipliers."""
        transform = OscTransform(
            vertical_velocity=2.0, horizontal_velocity=1.5, look_horizontal_velocity=0.8
        )
        action = torch.tensor([1, 2, 1, 1, 0], dtype=torch.long)

        result = transform(action)
        assert "axes" in result
        assert "buttons" in result

        assert result["axes"][OscAxes.Vertical] == 2.0  # 1.0 * 2.0
        assert result["axes"][OscAxes.Horizontal] == -1.5  # -1.0 * 1.5
        assert result["axes"][OscAxes.LookHorizontal] == 0.8  # 1.0 * 0.8
        assert result["buttons"][OscButtons.Jump] is True
        assert result["buttons"][OscButtons.Run] is False

    @pytest.mark.parametrize(
        "action,error_msg",
        [
            (torch.zeros((2, 5)), "Action tensor must be 1-dimensional"),
            (
                torch.zeros(4),
                f"Action tensor must have {len(OSC_ACTION_CHOICES)} elements",
            ),
            (torch.tensor([3, 0, 0, 0, 0]), "Invalid vertical movement action: 3"),
            (torch.tensor([0, 3, 0, 0, 0]), "Invalid horizontal movement action: 3"),
            (
                torch.tensor([0, 0, 3, 0, 0]),
                "Invalid look horizontal movement action: 3",
            ),
        ],
    )
    def test_call_invalid_input(self, action, error_msg):
        """Test transformation with invalid input tensors."""
        transform = OscTransform()

        with pytest.raises(ValueError, match=error_msg):
            transform(action)
