from pathlib import Path

import pytest
import yaml

from glove_chirality.config import ExtractionConfig


def test_default_config_loads():
    path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = ExtractionConfig.from_yaml(path)
    assert config.detector.backend == "belt_foreground"
    assert config.detector.adaptive_background is True
    assert config.event.make_square is True


def test_unknown_config_key_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"event": {"typo_setting": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="typo_setting"):
        ExtractionConfig.from_yaml(path)


def test_config_yaml_round_trip(tmp_path):
    config = ExtractionConfig()
    config.detector.roi = (0.1, 0.2, 0.8, 0.9)
    config.detector.color_distance_threshold = 31.5
    config.event.crop_padding = 0.2
    path = config.to_yaml(tmp_path / "nested" / "gui.yaml")
    loaded = ExtractionConfig.from_yaml(path)
    assert loaded.detector.roi == (0.1, 0.2, 0.8, 0.9)
    assert loaded.detector.color_distance_threshold == 31.5
    assert loaded.event.crop_padding == 0.2
