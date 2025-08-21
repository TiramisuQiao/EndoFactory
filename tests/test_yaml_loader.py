import json
from pathlib import Path
import polars as pl
import pytest

from endofactory.yaml_loader import YAMLConfigLoader
from endofactory.config import EndoFactoryConfig, ExportConfig, DatasetConfig


def test_create_config_from_scan_subdirs(tmp_path: Path):
    # Create dataset subdirs each with images and parquet
    ds1 = tmp_path / "dataset_a"
    ds2 = tmp_path / "dataset_b"
    (ds1 / "images").mkdir(parents=True)
    (ds2 / "imgs").mkdir(parents=True)
    # touch a few image files
    for p in [ds1/"images"/"a.jpg", ds1/"images"/"b.png", ds2/"imgs"/"c.jpg"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    # parquets (empty)
    for pq in [ds1/"meta.parquet", ds2/"meta.parquet"]:
        pl.DataFrame({"uuid": []}).write_parquet(pq)

    cfg = YAMLConfigLoader.create_config_from_scan(tmp_path)
    # Expect two datasets discovered
    assert len(cfg["datasets"]) == 2
    names = {d["name"] for d in cfg["datasets"]}
    assert names == {"dataset_a", "dataset_b"}


def test_create_config_from_scan_root_pairing(tmp_path: Path):
    # Root-level parquet and matching image dir by name
    pq = tmp_path / "endoscopy_vqa_v1.parquet"
    pl.DataFrame({"uuid": []}).write_parquet(pq)
    img_dir = tmp_path / "endoscopy_vqa_v1_images"
    img_dir.mkdir()
    (img_dir / "dummy.jpg").write_bytes(b"")

    cfg = YAMLConfigLoader.create_config_from_scan(tmp_path)
    assert len(cfg["datasets"]) == 1
    d = cfg["datasets"][0]
    assert Path(d["parquet_path"]).name == "endoscopy_vqa_v1.parquet"


def test_create_config_from_scan_no_pairs(tmp_path: Path):
    # Nothing to pair should raise
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        YAMLConfigLoader.create_config_from_scan(tmp_path)


def test_load_and_save_config_roundtrip(tmp_path: Path):
    # Build a minimal EndoFactoryConfig and roundtrip via YAML loader
    pq = tmp_path / "ds.parquet"
    pl.DataFrame({"uuid": ["u1"], "task": ["t"], "subtask": ["s"]}).write_parquet(pq)
    out_dir = tmp_path / "out"

    config = EndoFactoryConfig(
        datasets=[
            DatasetConfig(name="ds", image_path=tmp_path, parquet_path=pq, weight=1.0)
        ],
        columns=["uuid", "task", "subtask", "image_path"],
        export=ExportConfig(output_path=out_dir, format="parquet"),
        seed=123,
    )

    yaml_path = tmp_path / "cfg.yaml"
    YAMLConfigLoader.save_config(config, yaml_path)
    loaded = YAMLConfigLoader.load_config(yaml_path)

    assert loaded.export.output_path == out_dir
    assert loaded.datasets[0].name == "ds"
