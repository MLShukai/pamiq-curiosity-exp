import numpy as np
import pytest
import torch
from pamiq_vrchat.actuators import OscAction, OscAxes, OscButtons
from pytest_mock import MockerFixture

from exp.envs.vrchat import OSC_ACTION_CHOICES, OscTransform, create_env
from tests.helpers import parametrize_device


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


class TestCreateEnv:
    """Tests for the create_env function."""

    @pytest.fixture
    def mock_sensor(self, mocker: MockerFixture) -> None:
        mock = mocker.patch("exp.envs.vrchat.ImageSensor").return_value
        mock.read.return_value = np.random.randint(
            0, 256, (480, 620, 3), dtype=np.uint8
        )
        return mock

    @pytest.fixture
    def mock_actuator(self, mocker: MockerFixture) -> None:
        mock = mocker.patch("exp.envs.vrchat.SmoothOscActuator").return_value
        return mock

    def test_create_env_default_parameters(self, mocker: MockerFixture):
        """Test create_env with default parameters."""

        # Set up mock return values
        mock_sensor = mocker.patch("exp.envs.vrchat.ImageSensor")

        mock_actuator = mocker.patch("exp.envs.vrchat.SmoothOscActuator")

        # Call the function
        create_env()

        # Verify the mocks were called with correct default arguments
        mock_sensor.assert_called_once()
        mock_actuator.assert_called_once_with("127.0.0.1", 9000, delta_time=0.1)

    @parametrize_device
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_observe(self, mock_sensor, mock_actuator, device, dtype):
        env = create_env(image_size=(84, 84), device=device, dtype=dtype)

        obs = env.observe()
        mock_sensor.read.assert_called_once_with()
        assert obs.shape == (3, 84, 84)
        assert obs.mean().item() == pytest.approx(0.0, abs=0.01)
        assert obs.std().item() == pytest.approx(1.0, abs=0.01)
        assert obs.device == device
        assert obs.dtype == dtype

    @parametrize_device
    def test_affect(self, mock_sensor, mock_actuator, device):
        env = create_env(device=device, look_horizontal_velocity=0.7)

        env.affect(torch.ones(5, device=device))
        mock_actuator.operate.assert_called_once_with(
            OscAction(
                axes={
                    OscAxes.Vertical: 1.0,
                    OscAxes.Horizontal: 1.0,
                    OscAxes.LookHorizontal: 0.7,
                },
                buttons={
                    OscButtons.Jump: True,
                    OscButtons.Run: True,
                },
            )
        )
