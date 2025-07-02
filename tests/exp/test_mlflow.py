from pathlib import Path

import pytest

from exp.mlflow import flatten_config


class TestFlattenConfig:
    """Tests for the flatten_config function."""

    @pytest.mark.parametrize(
        "config,expected,parent_key,separator",
        [
            # Simple cases
            ({"a": 1, "b": 2}, {"a": 1, "b": 2}, "", "."),
            ([1, 2, "test"], {"0": 1, "1": 2, "2": "test"}, "", "."),
            ({}, {}, "", "."),
            ([], {}, "", "."),
            # Nested structures
            (
                {"model": {"hidden_dim": 128}, "training": {"lr": 0.001}},
                {"model.hidden_dim": 128, "training.lr": 0.001},
                "",
                ".",
            ),
            (
                [1, [2, 3], [4, [5, 6]]],
                {"0": 1, "1.0": 2, "1.1": 3, "2.0": 4, "2.1.0": 5, "2.1.1": 6},
                "",
                ".",
            ),
            # Mixed structures
            (
                {"model": {"layers": [64, 128]}, "data": [{"name": "train"}]},
                {"model.layers.0": 64, "model.layers.1": 128, "data.0.name": "train"},
                "",
                ".",
            ),
        ],
    )
    def test_flatten_config(self, config, expected, parent_key, separator):
        """Test flattening various config structures."""
        result = flatten_config(config, parent_key, separator)
        assert result == expected

    def test_flatten_with_mlflow_conversion(self):
        """Test that values are converted using convert_value_for_mlflow."""
        config = {"path": Path("/tmp")}
        result = flatten_config(config)

        assert isinstance(result["path"], str)
        assert "/tmp" in result["path"]
