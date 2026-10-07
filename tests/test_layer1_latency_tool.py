"""Regression checks for the latency report's pure statistics helper."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def latency_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "benchmark_layer1_latency.py"
    spec = importlib.util.spec_from_file_location("layer1_latency_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_latency_group_is_not_a_fabricated_zero(latency_tool):
    assert latency_tool.distribution_ms([]) is None


def test_constant_latency_reports_all_percentiles_and_sample_count(latency_tool):
    assert latency_tool.distribution_ms([5.0, 5.0, 5.0]) == {
        "samples": 3,
        "mean_ms": 5.0,
        "median_ms": 5.0,
        "p95_ms": 5.0,
        "p99_ms": 5.0,
        "max_ms": 5.0,
    }


def test_latency_distribution_does_not_modify_observations(latency_tool):
    values = [4.0, 1.0, 3.0, 2.0]
    result = latency_tool.distribution_ms(values)
    assert values == [4.0, 1.0, 3.0, 2.0]
    assert result["mean_ms"] == pytest.approx(2.5)
    assert result["median_ms"] == pytest.approx(2.5)
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["p99_ms"] == pytest.approx(3.97)
    assert result["max_ms"] == 4.0


def test_latency_digest_matches_standard_sha256(latency_tool, tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc")
    assert latency_tool.digest(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
