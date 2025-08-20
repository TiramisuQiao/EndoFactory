"""Configuration models for EndoFactory dataset construction."""

from typing import Dict, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator


class DatasetConfig(BaseModel):
    """Configuration for a single dataset source."""
    
    name: str = Field(..., description="Dataset name")
    image_path: Path = Field(..., description="Path to images directory")
    parquet_path: Path = Field(..., description="Path to parquet metadata file")
    weight: float = Field(default=1.0, description="Sampling weight for this dataset")
    columns: Optional[List[str]] = Field(default=None, description="Specific columns to extract")
    
    @validator('weight')
    def weight_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Weight must be positive')
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


class EndoFactoryConfig(BaseModel):
    """Main configuration for EndoFactory."""
    
    datasets: List[DatasetConfig] = Field(..., description="List of dataset configurations")
    columns: Optional[List[str]] = Field(default=None, description="Global columns to extract from all datasets")
    task_proportions: Optional[TaskProportionConfig] = Field(default=None)
    export: ExportConfig = Field(..., description="Export configuration")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    
    class Config:
        extra = "forbid"
