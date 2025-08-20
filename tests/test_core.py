import polars as pl
from pathlib import Path
from endofactory.core import EndoFactoryEngine


def test_load_and_mix(base_config):
    engine = EndoFactoryEngine(base_config)
    engine.load_datasets()

    # Ensure both datasets loaded
    assert set(engine.datasets.keys()) == {d.name for d in base_config.datasets}
    for name, df in engine.datasets.items():
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        # absolute_image_path should be created
        assert "absolute_image_path" in df.columns
        # source_dataset should be present
        assert "source_dataset" in df.columns

    mixed = engine.mix_datasets()
    assert mixed is not None
    assert len(mixed) > 0

    # Columns requested in config should exist (filled with null if missing originally)
    for col in base_config.columns:
        assert col in mixed.columns


def test_export_parquet(base_config, tmp_output_dir):
    engine = EndoFactoryEngine(base_config)
    engine.load_datasets()
    engine.mix_datasets()

    out_file = engine.export_dataset()
    assert out_file.exists()
    df = pl.read_parquet(out_file)
    assert len(df) > 0
