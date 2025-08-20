import os
from pathlib import Path
import tempfile
import shutil
import polars as pl
import pytest
from endofactory.config import EndoFactoryConfig, DatasetConfig, ExportConfig, TaskProportionConfig

BASE = Path(__file__).resolve().parents[1]
TEST_DATA = BASE / "test_data"

@pytest.fixture(scope="session")
def sample_datasets():
    """Provide paths to the prepared test datasets in test_data/."""
    d1 = {
        "name": "endoscopy_vqa_v1",
        "image_path": str(TEST_DATA / "endoscopy_vqa_v1" / "images"),
        "parquet_path": str(TEST_DATA / "endoscopy_vqa_v1" / "metadata.parquet"),
        "weight": 0.6,
    }
    d2 = {
        "name": "medical_vqa_v2",
        "image_path": str(TEST_DATA / "medical_vqa_v2" / "images"),
        "parquet_path": str(TEST_DATA / "medical_vqa_v2" / "metadata.parquet"),
        "weight": 0.4,
    }
    # Sanity check that files exist
    assert Path(d1["parquet_path"]).exists(), f"Missing {d1['parquet_path']}"
    assert Path(d2["parquet_path"]).exists(), f"Missing {d2['parquet_path']}"
    return d1, d2

@pytest.fixture()
def tmp_output_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out

@pytest.fixture()
def base_config(sample_datasets, tmp_output_dir):
    d1, d2 = sample_datasets
    cfg = EndoFactoryConfig(
        datasets=[
            DatasetConfig(**d1),
            DatasetConfig(**d2),
        ],
        columns=["uuid", "question", "answer", "task", "subtask", "category"],
        task_proportions=TaskProportionConfig(
            task_proportions={"classification": 0.5, "detection": 0.3, "segmentation": 0.2},
        ),
        export=ExportConfig(output_path=tmp_output_dir, format="parquet"),
        seed=42,
    )
    return cfg
