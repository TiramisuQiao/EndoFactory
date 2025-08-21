from pathlib import Path
import polars as pl
import pytest

from endofactory.core import EndoFactoryEngine
from endofactory.config import EndoFactoryConfig, DatasetConfig, ExportConfig, InputConfig


def make_engine_with_parquet(tmp_path: Path, data: pl.DataFrame, *, columns=None, weight=1.0):
    pq = tmp_path / "meta.parquet"
    data.write_parquet(pq)
    cfg = EndoFactoryConfig(
        datasets=[DatasetConfig(name="ds", image_path=tmp_path / "imgs", parquet_path=pq, weight=weight)],
        columns=columns,
        export=ExportConfig(output_path=tmp_path / "out", format="parquet"),
        seed=123,
    )
    return EndoFactoryEngine(cfg)


def test_load_single_dataset_fallback_columns(tmp_path: Path):
    # Trigger parquet columns fallback by requesting a missing column
    df = pl.DataFrame({"uuid": ["u1"], "task": ["t"], "subtask": ["s"]})
    engine = make_engine_with_parquet(tmp_path, df, columns=["uuid", "missing_col"])  # missing to trigger fallback
    engine.load_datasets()
    loaded = engine.datasets["ds"]
    # Current behavior: on missing columns, engine keeps available data without constructing extras
    assert "uuid" in loaded.columns


def test_load_single_dataset_no_source_raises(tmp_path: Path):
    # Neither parquet exists nor json_dir -> pydantic allows but path won't exist -> ValueError from engine
    cfg = EndoFactoryConfig(
        datasets=[DatasetConfig(name="ds", image_path=tmp_path, parquet_path=tmp_path / "no.parquet", weight=1.0)],
        columns=None,
        export=ExportConfig(output_path=tmp_path / "o", format="parquet"),
    )
    engine = EndoFactoryEngine(cfg)
    with pytest.raises(ValueError):
        engine.load_datasets()


def test_mix_datasets_branches(tmp_path: Path):
    # Build two datasets with sizes
    d1 = pl.DataFrame({"filename": ["a.jpg"] * 10})
    d2 = pl.DataFrame({"filename": ["b.jpg"] * 5})

    pq1 = tmp_path / "d1.parquet"; d1.write_parquet(pq1)
    pq2 = tmp_path / "d2.parquet"; d2.write_parquet(pq2)

    cfg = EndoFactoryConfig(
        datasets=[
            DatasetConfig(name="d1", image_path=tmp_path, parquet_path=pq1, weight=0.5),  # downsample to 5
            DatasetConfig(name="d2", image_path=tmp_path, parquet_path=pq2, weight=2.0),  # upsample to 10
        ],
        columns=["filename", "source_dataset", "image_path"],
        export=ExportConfig(output_path=tmp_path / "o", format="parquet"),
        seed=42,
    )
    engine = EndoFactoryEngine(cfg)
    engine.load_datasets()
    mixed = engine.mix_datasets()
    # Expect 5 + 10 = 15 rows
    assert len(mixed) == 15
    # Ensure both sources present
    assert set(mixed["source_dataset"].unique()) == {"d1", "d2"}


def test_mix_requires_load_first(tmp_path: Path):
    cfg = EndoFactoryConfig(
        datasets=[DatasetConfig(name="d", image_path=tmp_path, parquet_path=tmp_path/"x.parquet", weight=1.0)],
        columns=None,
        export=ExportConfig(output_path=tmp_path/"o", format="parquet"),
    )
    engine = EndoFactoryEngine(cfg)
    with pytest.raises(ValueError):
        engine.mix_datasets()


def test_task_proportions_application(tmp_path: Path):
    # Create data with tasks and subtasks, then apply proportions
    df = pl.DataFrame({
        "task": ["A"] * 10 + ["B"] * 10,
        "subtask": ["A1"] * 6 + ["A2"] * 4 + ["B1"] * 10,
    })
    engine = make_engine_with_parquet(tmp_path, df, columns=["task", "subtask", "image_path"])  # image_path added
    engine.load_datasets()
    # inject config proportions
    engine.config.task_proportions = type("TPC", (), {
        "task_proportions": {"A": 0.5, "B": 0.5},
        "subtask_proportions": {"A": {"A1": 0.5, "A2": 0.5}},
    })()
    mixed = engine.mix_datasets()
    # Verify both tasks are present and A's subtasks appear as requested
    tasks = set(mixed["task"].unique())
    assert {"A", "B"}.issubset(tasks)
    sub_for_a = set(mixed.filter(pl.col("task") == "A")["subtask"].unique())
    assert {"A1", "A2"}.issubset(sub_for_a)


def test_export_errors_and_stats(tmp_path: Path):
    # export_dataset without mixed should error
    cfg = EndoFactoryConfig(
        datasets=[DatasetConfig(name="d", image_path=tmp_path, parquet_path=tmp_path/"x.parquet", weight=1.0)],
        columns=None,
        export=ExportConfig(output_path=tmp_path/"o", format="parquet"),
    )
    engine = EndoFactoryEngine(cfg)
    with pytest.raises(ValueError):
        engine.export_dataset()

    # Prepare mixed and get stats including mixed_dataset path
    engine.mixed_dataset = pl.DataFrame({
        "source_dataset": ["d", "d"],
        "task": ["A", "B"],
    })
    stats = engine.get_dataset_stats()
    assert "mixed_dataset" in stats


def test_ingest_inputset_not_supported(tmp_path: Path):
    # Ingestion configured but different inputset
    cfg = EndoFactoryConfig(
        datasets=[],
        columns=None,
        export=ExportConfig(output_path=tmp_path/"o", format="parquet"),
        seed=0,
        input=InputConfig(inputset="ColonGPT", json_dir=tmp_path/"jj", images_root=tmp_path/"im"),
    )
    # Override to simulate unsupported
    cfg.input.inputset = "Other"
    engine = EndoFactoryEngine(cfg)
    assert engine.ingest_from_input() is None
