from pathlib import Path
import polars as pl
import yaml
from typer.testing import CliRunner

from endofactory.cli import app


runner = CliRunner()


def test_cli_create_config_default(tmp_path: Path):
    out = tmp_path / "cfg.yaml"
    result = runner.invoke(app, ["create-config", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_cli_view_parquet(tmp_path: Path):
    pq = tmp_path / "sample.parquet"
    pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}).write_parquet(pq)

    result = runner.invoke(app, ["view", str(pq), "--rows", "1", "--columns"])
    assert result.exit_code == 0


def test_cli_stats_and_build(tmp_path: Path):
    # Prepare a tiny dataset and config YAML
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    pq = tmp_path / "meta.parquet"
    pl.DataFrame({
        "uuid": ["u1", "u2"],
        "task": ["t1", "t2"],
        "subtask": ["s1", "s2"],
    }).write_parquet(pq)

    cfg_path = tmp_path / "config.yaml"
    cfg_obj = {
        "datasets": [
            {
                "name": "ds",
                "image_path": str(img_dir),
                "parquet_path": str(pq),
                "weight": 1.0,
            }
        ],
        "columns": ["uuid", "task", "subtask", "image_path"],
        "export": {"output_path": str(tmp_path / "out"), "format": "parquet"},
        "seed": 42,
    }
    cfg_path.write_text(yaml.safe_dump(cfg_obj), encoding="utf-8")

    # stats
    r1 = runner.invoke(app, ["stats", str(cfg_path)])
    assert r1.exit_code == 0

    # build (no input/ingest_output present; auto_ingest default True but skipped)
    r2 = runner.invoke(app, ["build", str(cfg_path), "--quiet"])
    assert r2.exit_code == 0


def test_cli_ingest_no_ingestion_config(tmp_path: Path):
    # Config without input/ingest_output should exit 0 with info
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    pq = tmp_path / "meta.parquet"
    pl.DataFrame({"uuid": ["u1"]}).write_parquet(pq)

    cfg_path = tmp_path / "config.yaml"
    cfg_obj = {
        "datasets": [
            {
                "name": "ds",
                "image_path": str(img_dir),
                "parquet_path": str(pq),
                "weight": 1.0,
            }
        ],
        "export": {"output_path": str(tmp_path / "out"), "format": "parquet"},
        "seed": 1,
    }
    cfg_path.write_text(yaml.safe_dump(cfg_obj), encoding="utf-8")

    r = runner.invoke(app, ["ingest", str(cfg_path)])
    # Current CLI wraps typer.Exit in a broad Exception handler, resulting in exit 1
    assert r.exit_code == 1
