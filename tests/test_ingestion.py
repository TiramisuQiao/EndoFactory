import json
from pathlib import Path

import polars as pl
import pytest

from endofactory.config import (
    DatasetConfig,
    EndoFactoryConfig,
    ExportConfig,
    InputConfig,
    IngestOutputConfig,
)
from endofactory.core import EndoFactoryEngine


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_parallel_json_ingestion(tmp_path: Path):
    # Prepare multiple small JSON files
    json_root = tmp_path / "jsons"
    _write_json(json_root / "a.json", [
        {"id": "ds1/img1.jpg", "image": "/abs/img1.jpg", "conversations": [{"role": "user", "content": "q1"}]},
        {"id": "ds1/img2.jpg", "image": "/abs/img2.jpg", "conversations": [{"role": "user", "content": "q2"}]},
    ])
    _write_json(json_root / "sub" / "b.json", {
        "id": "ds2/img3.jpg", "image": "/abs/img3.jpg", "conversations": [{"role": "user", "content": "q3"}]
    })

    cfg = EndoFactoryConfig(
        datasets=[
            DatasetConfig(
                name="colon",
                image_path=tmp_path / "images",
                json_dir=json_root,
                dataset_prefix=None,
                weight=1.0,
            )
        ],
        columns=["uuid", "id", "image", "conversations", "image_path"],
        export=ExportConfig(output_path=tmp_path / "out", format="parquet"),
        seed=42,
        num_workers=4,
    )

    engine = EndoFactoryEngine(cfg)
    engine.load_datasets()
    assert "colon" in engine.datasets
    df = engine.datasets["colon"]
    # 3 records from two files
    assert len(df) == 3
    # Ensure image_path is present (uuid may be absent for direct JSON ingestion)
    assert "image_path" in df.columns


def test_ingest_from_input_and_write_parquet(tmp_path: Path):
    json_root = tmp_path / "jroot"
    _write_json(json_root / "z.json", [
        {"id": "SUN/x1.jpg", "image": "/p/x1.jpg", "conversations": [{"role": "user", "content": "q"}]},
        {"id": "SUN/x2.jpg", "image": "/p/x2.jpg", "conversations": [{"role": "user", "content": "q"}]},
        {"id": "MOON/y.jpg", "image": "/p/y.jpg", "conversations": [{"role": "user", "content": "q"}]},
    ])

    cfg = EndoFactoryConfig(
        datasets=[],  # not used in this step
        columns=None,
        export=ExportConfig(output_path=tmp_path / "out", format="parquet"),
        seed=0,
        input=InputConfig(
            inputset="ColonGPT",
            json_dir=json_root,
            images_root=tmp_path / "imgs",
            dataset_prefix="SUN",
            auto_absolute_path=True,
        ),
        ingest_output=IngestOutputConfig(
            parquet_path=tmp_path / "ingested" / "colon.parquet",
            dataset_name="ColonGPT",
        ),
        num_workers=2,
    )

    engine = EndoFactoryEngine(cfg)
    df = engine.ingest_from_input()
    assert df is not None
    # Only SUN prefix remains
    assert all(str(x).startswith("SUN/") for x in df["id"].to_list())
    # Parquet should exist
    assert (tmp_path / "ingested" / "colon.parquet").exists()


def test_parquet_column_pushdown_and_image_path(tmp_path: Path):
    # Create a parquet file with many columns
    p = tmp_path / "meta.parquet"
    big = pl.DataFrame({
        "filename": ["a.jpg", "b.jpg", "c.jpg"],
        "task": ["A", "B", "A"],
        "subtask": ["A1", "B1", "A2"],
        "extra1": [1, 2, 3],
        "extra2": ["x", "y", "z"],
    })
    big.write_parquet(p)

    cfg = EndoFactoryConfig(
        datasets=[
            DatasetConfig(
                name="pds",
                image_path=tmp_path / "imgs",
                parquet_path=p,
                weight=1.0,
            )
        ],
        # Request only a subset to trigger pushdown
        columns=["filename", "task", "subtask", "source_dataset"],
        export=ExportConfig(output_path=tmp_path / "o", format="parquet"),
        seed=1,
        categorical_columns=["task", "subtask"],
    )

    engine = EndoFactoryEngine(cfg)
    engine.load_datasets()
    df = engine.datasets["pds"]
    # Expect requested columns present
    assert set(["filename", "task", "subtask", "source_dataset"]).issubset(df.columns)
    # Categorical types applied (check in a version-agnostic way)
    assert df.schema["task"] == pl.Categorical
    assert df.schema["subtask"] == pl.Categorical


def test_export_jsonl(tmp_path: Path):
    # Build a small dataset and export jsonl to ensure path works
    cfg = EndoFactoryConfig(
        datasets=[],
        columns=None,
        export=ExportConfig(output_path=tmp_path / "out", format="jsonl"),
        seed=0,
    )
    engine = EndoFactoryEngine(cfg)
    engine.mixed_dataset = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    outfile = engine.export_dataset()
    assert outfile.exists()
    assert outfile.suffix == ".jsonl"
