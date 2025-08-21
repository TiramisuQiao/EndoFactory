"""Configuration models for EndoFactory dataset construction."""

from typing import Dict, List, Optional, Union, Literal
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic import field_validator, model_validator
from pydantic import ConfigDict


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
    # Columns can be a list of strings (source names) or list of {source: target} mappings for rename
    columns: Optional[List[Union[str, Dict[str, str]]]] = Field(
        default=None,
        description="Specific columns to extract; support rename via list of one-key dicts, e.g., [{'image_path': 'path'}]",
    )

    @field_validator('weight')
    @classmethod
    def weight_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('Weight must be positive')
        return v

    @model_validator(mode='after')
    def _check_source(self):
        if self.parquet_path is None and self.json_dir is None:
            raise ValueError('Either parquet_path or json_dir must be provided')
        return self


class TaskProportionConfig(BaseModel):
    """Configuration for task and subtask proportions."""
    
    task_proportions: Dict[str, float] = Field(default_factory=dict)
    subtask_proportions: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def _check_proportions(self):
        # Do NOT enforce sum on task_proportions (kept for backward compatibility/tests)
        # Only check each subtask family sums to 1
        if self.subtask_proportions:
            for key, proportions in self.subtask_proportions.items():
                if isinstance(proportions, dict) and proportions:
                    total = sum(proportions.values())
                    if abs(total - 1.0) > 1e-6:
                        raise ValueError(f'Proportions for {key} must sum to 1.0, got {total}')
        return self


class ExportConfig(BaseModel):
    """Configuration for export settings."""
    
    output_path: Path = Field(..., description="Output directory path")
    format: str = Field(default="parquet", description="Export format: 'parquet', 'json', or 'jsonl'")
    include_absolute_paths: bool = Field(default=True, description="Include absolute image paths")

    @field_validator('format')
    @classmethod
    def format_must_be_valid(cls, v: str) -> str:
        if v not in ['parquet', 'json', 'jsonl']:
            raise ValueError('Format must be one of "parquet", "json", or "jsonl"')
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
    add_uuid: Optional[bool] = Field(
        default=False,
        description="If true, generate a deterministic uuid column from id during ingestion.",
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

    @model_validator(mode='after')
    def _infer_and_validate_paths(self):
        # Infer image_path_mode from auto_absolute_path if not provided
        if self.image_path_mode is None:
            self.image_path_mode = 'join_id' if (self.auto_absolute_path is True) else 'use_existing'
        # Validate images_root when required
        if self.image_path_mode == 'join_id' and self.images_root is None:
            raise ValueError("images_root is required when image_path_mode='join_id' (auto_absolute_path=True)")
        return self


class IngestOutputConfig(BaseModel):
    """Output configuration for the ingestion step producing an intermediate parquet."""

    parquet_path: Path = Field(..., description="Output parquet path for the ingested data")
    dataset_name: str = Field('ColonGPT', description="Name to tag this ingested dataset")


class EndoFactoryConfig(BaseModel):
    """Main configuration for EndoFactory."""
    
    datasets: List[DatasetConfig] = Field(..., description="List of dataset configurations")
    # Global columns with same semantics as DatasetConfig.columns
    columns: Optional[List[Union[str, Dict[str, str]]]] = Field(
        default=None,
        description="Global columns to extract; support rename via list of one-key dicts.",
    )
    task_proportions: Optional[TaskProportionConfig] = Field(default=None)
    export: ExportConfig = Field(..., description="Export configuration")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    # Convert selected columns to list type (wrap scalar into single-element list)
    listify_columns: Optional[List[str]] = Field(
        default=None,
        description="Columns to convert to List type by wrapping non-null scalar values into a single-element list. Applied after rename.",
    )
    # Optional ingestion configs
    input: Optional[InputConfig] = Field(default=None, description="Optional raw input ingestion configuration")
    ingest_output: Optional[IngestOutputConfig] = Field(default=None, description="Optional ingestion output configuration")
    # Performance-related options
    num_workers: Optional[int] = Field(
        default=None,
        description="Optional number of worker threads for parallel JSON reading. If None, uses a sensible default.",
    )
    categorical_columns: Optional[List[str]] = Field(
        default=None,
        description="Optional list of column names to cast to categorical to reduce memory footprint.",
    )

    # Pydantic v2 config
    model_config = ConfigDict(extra='forbid')
