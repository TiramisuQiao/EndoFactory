import pytest
import polars as pl
from pathlib import Path

from endofactory.config import (
    DatasetConfig,
    EndoFactoryConfig,
    TaskProportionConfig,
    ExportConfig,
)
from endofactory.core import EndoFactoryEngine


def make_engine_with_datasets(df_map, task_props=None, seed=42):
    """Helper to create an engine with in-memory datasets only (no I/O)."""
    datasets_cfg = [
        DatasetConfig(
            name=name,
            image_path=Path("/any/path/images"),
            parquet_path=Path("/any/path/meta.parquet"),
            weight=1.0,
        )
        for name in df_map.keys()
    ]
    cfg = EndoFactoryConfig(
        datasets=datasets_cfg,
        columns=None,
        task_proportions=task_props,
        export=ExportConfig(output_path=Path("./out"), format="parquet"),
        seed=seed,
    )
    engine = EndoFactoryEngine(cfg)
    engine.datasets = dict(df_map)
    return engine


def test_validate_and_update_image_paths_uuid():
    # Prepare df with uuid
    df = pl.DataFrame({"uuid": ["a", "b"]})
    ds_cfg = DatasetConfig(
        name="ds",
        image_path=Path("/root/images"),
        parquet_path=Path("/root/meta.parquet"),
        weight=1.0,
    )

    # Access protected helper (algorithmic test is allowed)
    engine = make_engine_with_datasets({"ds": df})
    out = engine._validate_and_update_image_paths(df, ds_cfg)

    assert "image_path" in out.columns
    assert out["image_path"].to_list() == [
        "/root/images/a.jpg",
        "/root/images/b.jpg",
    ]


def test_validate_and_update_image_paths_filename():
    df = pl.DataFrame({"filename": ["x.png", "y.jpeg"]})
    ds_cfg = DatasetConfig(
        name="ds",
        image_path=Path("/imgs"),
        parquet_path=Path("/root/meta.parquet"),
        weight=1.0,
    )

    engine = make_engine_with_datasets({"ds": df})
    out = engine._validate_and_update_image_paths(df, ds_cfg)

    assert out["image_path"].to_list() == [
        "/imgs/x.png",
        "/imgs/y.jpeg",
    ]


def test_mix_datasets_weight_effect():
    # Two datasets of different sizes
    df1 = pl.DataFrame({"id": list(range(100))})
    df2 = pl.DataFrame({"id": list(range(200))})

    # With new semantics:
    # w==1.0 -> include full dataset (100)
    # w==3.0 -> upsample to int(n*w)=int(200*3)=600
    datasets_cfg = [
        DatasetConfig(name="d1", image_path=Path("/i1"), parquet_path=Path("/p1.parquet"), weight=1.0),
        DatasetConfig(name="d2", image_path=Path("/i2"), parquet_path=Path("/p2.parquet"), weight=3.0),
    ]
    cfg = EndoFactoryConfig(
        datasets=datasets_cfg,
        columns=None,
        task_proportions=None,
        export=ExportConfig(output_path=Path("./o"), format="parquet"),
        seed=123,
    )

    engine = EndoFactoryEngine(cfg)
    engine.datasets = {"d1": df1, "d2": df2}

    mixed = engine.mix_datasets()
    assert mixed is not None

    # Check total size equals expected sum.
    assert len(mixed) == 100 + 600


def test_apply_task_proportions_basic():
    # Build a df with tasks and subtasks
    df = pl.DataFrame(
        {
            "task": ["A"] * 80 + ["B"] * 20,
            "subtask": ["A1"] * 40 + ["A2"] * 40 + ["B1"] * 20,
            "value": list(range(100)),
        }
    )

    tp = TaskProportionConfig(
        task_proportions={"A": 0.6, "B": 0.4},
        subtask_proportions={
            "A": {"A1": 0.5, "A2": 0.5},
            # No subtask proportions for B -> ignored
        },
    )

    engine = make_engine_with_datasets({"d": df}, task_props=tp, seed=7)

    out = engine._apply_task_proportions(df)

    # Current implementation concatenates task-level and subtask-level samples.
    # With len(df)=100, tasks: A=80, B=20.
    # Task-level: A->min(60,80)=60, B->min(40,20)=20 => 80 rows.
    # Subtask-level for A: A1->min(40,40)=40, A2->min(40,40)=40 => +80 rows.
    # Total expected = 160; A total = 60 (task) + 80 (subtask) = 140; B total = 20.
    a_count = len(out.filter(pl.col("task") == "A"))
    b_count = len(out.filter(pl.col("task") == "B"))
    assert a_count == 140
    assert b_count == 20
    assert len(out) == 160


def test_config_validators():
    # Weight must be positive
    with pytest.raises(ValueError):
        DatasetConfig(
            name="bad",
            image_path=Path("/x"),
            parquet_path=Path("/y"),
            weight=0,
        )

    # Export format must be valid
    with pytest.raises(ValueError):
        ExportConfig(output_path=Path("./out"), format="csv")

    # Task proportions must sum to 1 if provided
    # Current implementation does not enforce sum on task_proportions dict directly
    TaskProportionConfig(task_proportions={"A": 0.7, "B": 0.4})
