"""Core data processing engine for EndoFactory using Polars."""

import uuid
from pathlib import Path
from typing import List, Dict, Optional, Union
import polars as pl
import json
from .config import EndoFactoryConfig, DatasetConfig


class EndoFactoryEngine:
    """Main engine for processing and mixing EndoVQA datasets."""
    
    def __init__(self, config: EndoFactoryConfig):
        self.config = config
        self.datasets: Dict[str, pl.DataFrame] = {}
        self.mixed_dataset: Optional[pl.DataFrame] = None
        
    def load_datasets(self) -> None:
        """Load all configured datasets into memory."""
        for dataset_config in self.config.datasets:
            df = self._load_single_dataset(dataset_config)
            self.datasets[dataset_config.name] = df
            
    def _load_single_dataset(self, dataset_config: DatasetConfig) -> pl.DataFrame:
        """Load a single dataset from parquet file."""
        # Load parquet file
        df = pl.read_parquet(dataset_config.parquet_path)
        
        # Use global columns configuration or dataset-specific columns
        columns_to_extract = self.config.columns or dataset_config.columns
        
        # Filter and standardize columns if specified
        if columns_to_extract:
            # Create a new dataframe with only the requested columns
            # If a column doesn't exist, fill with null values
            column_exprs = []
            for col in columns_to_extract:
                if col in df.columns:
                    column_exprs.append(pl.col(col))
                else:
                    # Create null column for missing columns
                    column_exprs.append(pl.lit(None).alias(col))
            df = df.select(column_exprs)
        
        # Add dataset name column
        df = df.with_columns(pl.lit(dataset_config.name).alias("source_dataset"))
        
        # Validate image paths and update absolute paths
        df = self._validate_and_update_image_paths(df, dataset_config)
        
        return df
    
    def _validate_and_update_image_paths(self, df: pl.DataFrame, dataset_config: DatasetConfig) -> pl.DataFrame:
        """Validate image existence and create absolute paths."""
        image_dir = Path(dataset_config.image_path)
        
        # Create absolute image paths
        if 'uuid' in df.columns:
            df = df.with_columns(
                (pl.lit(str(image_dir)) + "/" + pl.col("uuid") + ".jpg").alias("absolute_image_path")
            )
        elif 'filename' in df.columns:
            df = df.with_columns(
                (pl.lit(str(image_dir)) + "/" + pl.col("filename")).alias("absolute_image_path")
            )
        
        return df
    
    def mix_datasets(self) -> pl.DataFrame:
        """Mix datasets according to configured proportions."""
        if not self.datasets:
            raise ValueError("No datasets loaded. Call load_datasets() first.")
        
        mixed_dfs = []
        
        # Calculate total weight
        total_weight = sum(config.weight for config in self.config.datasets)
        
        for dataset_config in self.config.datasets:
            df = self.datasets[dataset_config.name]
            
            # Calculate sample size based on weight
            proportion = dataset_config.weight / total_weight
            sample_size = int(len(df) * proportion)
            
            if sample_size > 0:
                # Sample with seed for reproducibility
                sampled_df = df.sample(n=min(sample_size, len(df)), seed=self.config.seed)
                mixed_dfs.append(sampled_df)
        
        # Combine all datasets
        if mixed_dfs:
            self.mixed_dataset = pl.concat(mixed_dfs, how="diagonal")
            
            # Apply task/subtask proportions if configured
            if self.config.task_proportions:
                self.mixed_dataset = self._apply_task_proportions(self.mixed_dataset)
            
            # Shuffle the final dataset
            self.mixed_dataset = self.mixed_dataset.sample(fraction=1.0, seed=self.config.seed)
        
        return self.mixed_dataset
    
    def _apply_task_proportions(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply task and subtask proportion constraints."""
        if not self.config.task_proportions:
            return df
        
        result_dfs = []
        
        # Handle task proportions
        if self.config.task_proportions.task_proportions and 'task' in df.columns:
            for task, proportion in self.config.task_proportions.task_proportions.items():
                task_df = df.filter(pl.col("task") == task)
                if len(task_df) > 0:
                    target_size = int(len(df) * proportion)
                    sampled_task_df = task_df.sample(n=min(target_size, len(task_df)), seed=self.config.seed)
                    result_dfs.append(sampled_task_df)
        
        # Handle subtask proportions
        if self.config.task_proportions.subtask_proportions and 'subtask' in df.columns:
            for task, subtask_props in self.config.task_proportions.subtask_proportions.items():
                task_df = df.filter(pl.col("task") == task) if 'task' in df.columns else df
                
                for subtask, proportion in subtask_props.items():
                    subtask_df = task_df.filter(pl.col("subtask") == subtask)
                    if len(subtask_df) > 0:
                        target_size = int(len(task_df) * proportion)
                        sampled_subtask_df = subtask_df.sample(n=min(target_size, len(subtask_df)), seed=self.config.seed)
                        result_dfs.append(sampled_subtask_df)
        
        if result_dfs:
            return pl.concat(result_dfs, how="vertical_relaxed")
        return df
    
    def export_dataset(self) -> Path:
        """Export the mixed dataset to specified format."""
        if self.mixed_dataset is None:
            raise ValueError("No mixed dataset available. Call mix_datasets() first.")
        
        output_path = Path(self.config.export.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.config.export.format == "parquet":
            output_file = output_path / "endovqa_dataset.parquet"
            self.mixed_dataset.write_parquet(output_file)
        
        elif self.config.export.format == "jsonl":
            output_file = output_path / "endovqa_dataset.jsonl"
            self._export_jsonl(output_file)
        
        return output_file
    
    def _export_jsonl(self, output_file: Path) -> None:
        """Export dataset to JSONL format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in self.mixed_dataset.iter_rows(named=True):
                json.dump(row, f, ensure_ascii=False)
                f.write('\n')
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about loaded datasets."""
        stats = {}
        
        for name, df in self.datasets.items():
            stats[name] = {
                "total_samples": len(df),
                "columns": df.columns,
                "tasks": df["task"].unique().to_list() if "task" in df.columns else [],
                "subtasks": df["subtask"].unique().to_list() if "subtask" in df.columns else []
            }
        
        if self.mixed_dataset is not None:
            stats["mixed_dataset"] = {
                "total_samples": len(self.mixed_dataset),
                "source_distribution": self.mixed_dataset["source_dataset"].value_counts().to_dict() if "source_dataset" in self.mixed_dataset.columns else {}
            }
        
        return stats
