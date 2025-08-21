"""Core data processing engine for EndoFactory using Polars."""

import uuid
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
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
        """Load a single dataset from parquet file or directly from JSON."""
        # Load from parquet if available, otherwise ingest from JSON directly
        if dataset_config.parquet_path and Path(dataset_config.parquet_path).exists():
            df = pl.read_parquet(dataset_config.parquet_path)
        elif dataset_config.json_dir:
            df = self._ingest_dataset_from_json(dataset_config)
        else:
            raise ValueError(f"Dataset {dataset_config.name}: neither parquet_path exists nor json_dir provided")
        
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
        
        # Add dataset name column only if user wants it (in columns) or no column filter is set
        if columns_to_extract is None or (isinstance(columns_to_extract, list) and "source_dataset" in columns_to_extract):
            df = df.with_columns(pl.lit(dataset_config.name).alias("source_dataset"))
        
        # Validate image paths and update absolute paths
        df = self._validate_and_update_image_paths(df, dataset_config)
        
        return df

    # ============ Ingestion (ColonGPT) ============
    def ingest_from_input(self) -> Optional[pl.DataFrame]:
        """If input ingestion is configured, scan JSON and build a parquet.

        Returns the ingested DataFrame if performed, otherwise None.
        """
        if not self.config.input or not self.config.ingest_output:
            return None

        input_cfg = self.config.input
        ingest_out = self.config.ingest_output

        if input_cfg.inputset != 'ColonGPT':
            # Currently only ColonGPT is supported as requested
            return None

        json_dir = Path(input_cfg.json_dir)
        if not json_dir.exists():
            raise FileNotFoundError(f"JSON dir not found: {json_dir}")

        records: List[Dict[str, Any]] = []
        # Scan all json files under json_dir (recursive)
        for p in json_dir.rglob('*.json'):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to read JSON {p}: {e}")

            # Each file may contain a list (recommended) or a single object
            items = data if isinstance(data, list) else [data]
            for it in items:
                # Required fields from example: id, image, conversations
                if not all(k in it for k in ('id', 'image', 'conversations')):
                    # Skip invalid entries
                    continue

                rec: Dict[str, Any] = {
                    'id': it['id'],
                    'image': it['image'],
                    'conversations': it['conversations'],
                }

                # Optional derive image_path
                if input_cfg.image_path_mode == 'join_id':
                    # id is a relative path; join with images_root
                    assert input_cfg.images_root is not None
                    rec['image_path'] = str(Path(input_cfg.images_root) / it['id'])
                elif input_cfg.image_path_mode == 'use_existing':
                    # image field is already an absolute path
                    # keep as-is; do not add image_path
                    pass

                # Add a stable uuid for this row
                rec['uuid'] = str(uuid.uuid5(uuid.NAMESPACE_URL, rec['id']))

                records.append(rec)

        if not records:
            raise ValueError("No valid ColonGPT records found from input JSONs.")

        df = pl.from_dicts(records)

        # Optional filter by dataset prefix (id startswith '<prefix>/')
        if input_cfg.dataset_prefix:
            pref = input_cfg.dataset_prefix.rstrip('/') + '/'
            if 'id' in df.columns:
                df = df.filter(pl.col('id').str.starts_with(pref))

        # Save to parquet
        out_path = Path(ingest_out.parquet_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)

        # Optionally register as a dataset source for subsequent build
        # Users may already list datasets referencing this parquet; we don't mutate config here.
        return df
    
    def _ingest_dataset_from_json(self, dataset_config: DatasetConfig) -> pl.DataFrame:
        """Directly ingest a dataset from JSON files without intermediate parquet."""
        json_dir = Path(dataset_config.json_dir)
        if not json_dir.exists():
            raise FileNotFoundError(f"JSON dir not found: {json_dir}")

        records: List[Dict[str, Any]] = []
        # Scan all json files under json_dir (recursive)
        for p in json_dir.rglob('*.json'):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to read JSON {p}: {e}")

            # Each file may contain a list (recommended) or a single object
            items = data if isinstance(data, list) else [data]
            for it in items:
                # Required fields from example: id, image, conversations
                if not all(k in it for k in ('id', 'image', 'conversations')):
                    # Skip invalid entries
                    continue

                rec: Dict[str, Any] = {
                    'id': it['id'],
                    'image': it['image'],
                    'conversations': it['conversations'],
                }

                # Optional derive image_path
                if dataset_config.auto_absolute_path:
                    # id is a relative path; join with images_root
                    rec['image_path'] = str(Path(dataset_config.image_path) / it['id'])
                elif 'image' in it and Path(it['image']).is_absolute():
                    # image field is already an absolute path
                    rec['image_path'] = it['image']

                # Add a stable uuid for this row
                rec['uuid'] = str(uuid.uuid5(uuid.NAMESPACE_URL, rec['id']))

                records.append(rec)

        if not records:
            raise ValueError(f"No valid ColonGPT records found from JSON dir: {json_dir}")

        df = pl.from_dicts(records)

        # Optional filter by dataset prefix (id startswith '<prefix>/')
        if dataset_config.dataset_prefix:
            pref = dataset_config.dataset_prefix.rstrip('/') + '/'
            if 'id' in df.columns:
                df = df.filter(pl.col('id').str.starts_with(pref))

        return df
    
    def _validate_and_update_image_paths(self, df: pl.DataFrame, dataset_config: DatasetConfig) -> pl.DataFrame:
        """Validate image existence and create absolute paths."""
        image_dir = Path(dataset_config.image_path)
        
        # If dataset already provides an absolute image_path column (e.g., from ingestion), keep it
        if 'image_path' in df.columns:
            return df

        # Create absolute image paths from available identifiers -> standardize to 'image_path'
        if 'uuid' in df.columns:
            df = df.with_columns(
                (pl.lit(str(image_dir)) + "/" + pl.col("uuid") + ".jpg").alias("image_path")
            )
        elif 'filename' in df.columns:
            df = df.with_columns(
                (pl.lit(str(image_dir)) + "/" + pl.col("filename")).alias("image_path")
            )
        
        return df
    
    def mix_datasets(self) -> pl.DataFrame:
        """Mix datasets according to configured weights.

        New semantics:
        - weight == 1.0: include the whole dataset once (no downsampling) -> total equals the sum of sizes.
        - weight < 1.0: downsample without replacement to floor(weight * N).
        - weight > 1.0: upsample to floor(weight * N) using sampling with replacement for the extra part.
        """
        if not self.datasets:
            raise ValueError("No datasets loaded. Call load_datasets() first.")
        
        mixed_dfs: List[pl.DataFrame] = []

        for dataset_config in self.config.datasets:
            df = self.datasets[dataset_config.name]
            n = len(df)
            w = dataset_config.weight

            if w == 1 or abs(w - 1.0) < 1e-9:
                mixed_dfs.append(df)
                continue

            target = int(n * w)
            if target <= 0:
                continue

            if w < 1.0:
                # downsample without replacement
                k = min(target, n)
                sampled_df = df.sample(n=k, with_replacement=False, seed=self.config.seed)
                mixed_dfs.append(sampled_df)
            else:
                # upsample: full copies + partial remainder, with replacement
                full = target // n
                rem = target % n
                parts = []
                if full > 0:
                    # repeat full copies
                    parts.append(pl.concat([df] * full, how="diagonal"))
                if rem > 0:
                    parts.append(df.sample(n=rem, with_replacement=True, seed=self.config.seed))
                sampled_df = pl.concat(parts, how="diagonal") if parts else df.head(0)
                mixed_dfs.append(sampled_df)
        
        # Combine all datasets
        if mixed_dfs:
            self.mixed_dataset = pl.concat(mixed_dfs, how="diagonal")
            
            # Apply task/subtask proportions if configured and applicable
            # If inputset is ColonGPT (which has no task/subtask), skip applying proportions
            if (self.config.task_proportions 
                and not (self.config.input and self.config.input.inputset == 'ColonGPT')):
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
