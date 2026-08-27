from pathlib import Path

import pytest
import yaml

from glove_chirality.config import ExtractionConfig


def test_default_config_loads():
    path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = ExtractionConfig.from_yaml(path)
    assert config.detector.backend == "dark_contour"
    assert config.event.make_square is True


def test_unknown_config_key_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"event": {"typo_setting": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="typo_setting"):
        ExtractionConfig.from_yaml(path)
