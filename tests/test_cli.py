from pathlib import Path
import polars as pl
import pytest
from typer.testing import CliRunner
from endofactory.cli import app
from endofactory.yaml_loader import YAMLConfigLoader

runner = CliRunner()


def test_cli_create_config(tmp_path):
    out_path = tmp_path / "cfg.yaml"
    result = runner.invoke(app, ["create-config", "--output", str(out_path)])
    assert result.exit_code == 0
    assert out_path.exists()


def test_cli_build_and_stats(base_config, tmp_path):
    # Save config to disk
    cfg_file = tmp_path / "cfg.yaml"
    YAMLConfigLoader.save_config(base_config, cfg_file)

    # Build
    result_build = runner.invoke(app, ["build", str(cfg_file), "--verbose"])  # noqa: FBT003
    assert result_build.exit_code == 0

    # Output parquet should exist
    out_parquet = Path(base_config.export.output_path) / "endovqa_dataset.parquet"
    assert out_parquet.exists()

    # Stats
    result_stats = runner.invoke(app, ["stats", str(cfg_file)])
    assert result_stats.exit_code == 0

    # View
    result_view = runner.invoke(app, ["view", str(out_parquet), "--rows", "5", "--columns"])  # noqa: FBT003
    assert result_view.exit_code == 0
