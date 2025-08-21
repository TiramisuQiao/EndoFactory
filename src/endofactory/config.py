"""Configuration models for EndoFactory dataset construction."""

from typing import Dict, List, Optional, Union, Literal
from pathlib import Path
from pydantic import BaseModel, Field, validator


class DatasetConfig(BaseModel):
    """Configuration for a single dataset source."""
    
    name: str = Field(..., description="Dataset name")
    image_path: Path = Field(..., description="Path to images directory")
    # Either parquet_path OR json_dir + dataset_prefix for direct ingestion
    parquet_path: Optional[Path] = Field(default=None, description="Path to parquet metadata file")
    json_dir: Optional[Path] = Field(default=None, description="Directory containing JSON files for direct ingestion")
    dataset_prefix: Optional[str] = Field(default=None, description="Dataset prefix filter for JSON ingestion (e.g., 'SUN')")
    auto_absolute_path: Optional[bool] = Field(default=True, description="Generate absolute paths from images_root + id")
    weight: float = Field(default=1.0, description="Sampling weight for this dataset")
    columns: Optional[List[str]] = Field(default=None, description="Specific columns to extract")
    
    @validator('weight')
    def weight_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Weight must be positive')
        return v
    
    @validator('json_dir', always=True)
    def validate_source(cls, v, values):
        parquet_path = values.get('parquet_path')
        if v is None and parquet_path is None:
            raise ValueError('Either parquet_path or json_dir must be provided')
        return v


class TaskProportionConfig(BaseModel):
    """Configuration for task and subtask proportions."""
    
    task_proportions: Dict[str, float] = Field(default_factory=dict)
    subtask_proportions: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    @validator('task_proportions', 'subtask_proportions')
    def proportions_must_sum_to_one(cls, v):
        if isinstance(v, dict) and v:
            for key, proportions in v.items():
                if isinstance(proportions, dict):
                    total = sum(proportions.values())
                    if abs(total - 1.0) > 1e-6:
                        raise ValueError(f'Proportions for {key} must sum to 1.0, got {total}')
                elif isinstance(v, dict) and key == 'task_proportions':
                    total = sum(v.values())
                    if abs(total - 1.0) > 1e-6:
                        raise ValueError(f'Task proportions must sum to 1.0, got {total}')
        return v


class ExportConfig(BaseModel):
    """Configuration for export settings."""
    
    output_path: Path = Field(..., description="Output directory path")
    format: str = Field(default="parquet", description="Export format: 'parquet' or 'jsonl'")
    include_absolute_paths: bool = Field(default=True, description="Include absolute image paths")
    
    @validator('format')
    def format_must_be_valid(cls, v):
        if v not in ['parquet', 'jsonl']:
            raise ValueError('Format must be either "parquet" or "jsonl"')
        return v


class InputConfig(BaseModel):
    """Configuration for raw input ingestion stage (e.g., ColonGPT JSON + images)."""

    inputset: Literal['ColonGPT'] = Field(..., description="Type of input set.")
    json_dir: Path = Field(..., description="Directory containing JSON files to scan")
    images_root: Optional[Path] = Field(
        default=None,
        description="Root directory for images. Required when image_path_mode='join_id'",
    )
    dataset_prefix: Optional[str] = Field(
        default=None,
        description="Optional dataset prefix to filter records by id (e.g., 'SUN' to include ids starting with 'SUN/')",
    )
    # Preferred boolean switch
    auto_absolute_path: Optional[bool] = Field(
        default=True,
        description="If true, join images_root with record 'id' to form image_path; if false, use existing absolute path in JSON",
    )
    # Backward-compatible mode; will be inferred from auto_absolute_path if not provided
    image_path_mode: Optional[Literal['join_id', 'use_existing']] = Field(
        default=None,
        description="Deprecated: prefer auto_absolute_path. If provided, overrides auto_absolute_path.",
    )

    @validator('image_path_mode', always=True)
    def infer_mode_from_auto(cls, v, values):
        # If mode is explicitly provided, keep it; otherwise infer from auto_absolute_path
        if v is not None:
            return v
        auto = values.get('auto_absolute_path', True)
        return 'join_id' if auto else 'use_existing'

    @validator('images_root', always=True)
    def validate_images_root(cls, v, values):
        mode = values.get('image_path_mode')
        if mode == 'join_id' and v is None:
            raise ValueError("images_root is required when image_path_mode='join_id' (auto_absolute_path=True)")
        return v


class IngestOutputConfig(BaseModel):
    """Output configuration for the ingestion step producing an intermediate parquet."""

    parquet_path: Path = Field(..., description="Output parquet path for the ingested data")
    dataset_name: str = Field('ColonGPT', description="Name to tag this ingested dataset")


class EndoFactoryConfig(BaseModel):
    """Main configuration for EndoFactory."""
    
    datasets: List[DatasetConfig] = Field(..., description="List of dataset configurations")
    columns: Optional[List[str]] = Field(default=None, description="Global columns to extract from all datasets")
    task_proportions: Optional[TaskProportionConfig] = Field(default=None)
    export: ExportConfig = Field(..., description="Export configuration")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    # Optional ingestion configs
    input: Optional[InputConfig] = Field(default=None, description="Optional raw input ingestion configuration")
    ingest_output: Optional[IngestOutputConfig] = Field(default=None, description="Optional ingestion output configuration")
    
    class Config:
        extra = "forbid"
