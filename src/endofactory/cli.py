"""Command-line interface for EndoFactory."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from .yaml_loader import YAMLConfigLoader
from .core import EndoFactoryEngine
import logging
import os
import shutil

app = typer.Typer(help="EndoFactory: Revolutionary EndoVQA dataset construction tool")
console = Console()


@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging and rich tracebacks"),
):
    """Global options for the EndoFactory CLI."""
    if debug:
        # Enable DEBUG logging for the application
        os.environ["ENDOFACTORY_DEBUG"] = "1"
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
        try:
            from rich.traceback import install as _rt_install
            _rt_install(show_locals=True)
        except Exception:
            # Rich may not be available in some environments; continue without it
            pass
        rprint("🐞 [bold yellow]Debug mode enabled[/bold yellow]")

@app.command()
def create_config(
    output: Path = typer.Option("config.yaml", help="Output path for configuration file"),
    scan_dir: Optional[Path] = typer.Option(None, help="Optional: scan a directory to auto-detect datasets (images + parquet)"),
    colon_gpt: bool = typer.Option(False, help="Generate a ColonGPT-style input config and scaffold fake data under test_data"),
    split: str = typer.Option("train", help="Split for ColonGPT: train|val|test"),
    auto_absolute_path: bool = typer.Option(True, help="When ColonGPT mode: if true, join images_root with id to build image_path")
):
    """Create a configuration file.
    
    Modes:
    - Default: writes an example configuration.
    - --scan-dir: auto-detect datasets under the given directory.
    - --colon-gpt: scaffold ColonGPT-style directories and JSON/images, and create an ingestion config.
    """
    import yaml
    from pathlib import Path as _Path

    # Special mode: ColonGPT + scan_dir -> generate a single merged config with all datasets (no scaffold)
    if colon_gpt and scan_dir is not None:
        split_l = split.lower()
        if split_l not in {"train", "val", "test"}:
            raise typer.BadParameter("split must be one of: train, val, test")
        json_split_name = {"train": "train", "val": "test", "test": "test"}[split_l]

        # Detect dataset names
        ds_names = _detect_dataset_prefixes(scan_dir)
        if not ds_names:
            rprint(f"ℹ️  No dataset prefixes detected under {scan_dir}. Expected Positive-images/* or JSON ids like 'SUN/Train/...' ")
            raise typer.Exit(1)

        datasets_cfg = []
        for ds in sorted(ds_names):
            datasets_cfg.append({
                "name": ds,
                "image_path": str((scan_dir / 'Positive-images').resolve()),
                "json_dir": str((scan_dir / 'Json-file' / json_split_name).resolve()),
                "dataset_prefix": ds,
                "auto_absolute_path": bool(auto_absolute_path),
                "weight": 1.0,
            })

        # Single merged config; omit input/ingest_output because ingestion is per-dataset
        merged = {
            "datasets": datasets_cfg,
            "columns": ["id", "image", "conversations", "image_path"],
            "export": {
                "output_path": "./output",
                "format": "parquet",
                "include_absolute_paths": True,
            },
            "seed": 42,
        }

        with open(output, 'w', encoding='utf-8') as f:
            yaml.dump(merged, f, default_flow_style=False, allow_unicode=True)

        rprint(f"✅ Merged config created at: {output}")
        rprint("📦 Datasets included: " + ", ".join(sorted(ds_names)))
        rprint("💡 Next: run 'endofactory build' to directly process and mix datasets.")
        return

    if colon_gpt:
        # Build ColonGPT-style config and scaffold
        repo_root = _Path(__file__).resolve().parents[2]
        td = repo_root / "tests" / "data"
        json_root = td / "Json-file"
        img_root = td / "Positive-images"
        ingest_out = td / "ingested.parquet"

        # Split mapping rule: user requested val -> Json-file/test, images -> .../Test
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise typer.BadParameter("split must be one of: train, val, test")
        json_split_dir = {
            "train": json_root / "train",
            "val": json_root / "test",   # special mapping per user requirement
            "test": json_root / "test",
        }[split]

        # Scaffold fake data (two datasets: SUN and LUNA, with Train/Test images)
        _scaffold_colon_gpt_fake(td, json_root, img_root)

        # Create ColonGPT-style config
        config_dict = {
            "input": {
                "inputset": "ColonGPT",
                "json_dir": str(json_split_dir.resolve()),
                "auto_absolute_path": bool(auto_absolute_path),
                "images_root": str(img_root.resolve()),
                # For 'val', we still read Json-file/test and images Test according to rule
            },
            "ingest_output": {
                "parquet_path": str(ingest_out.resolve()),
                "dataset_name": "ColonGPT",
            },
            "datasets": [
                {
                    "name": "ColonGPT",
                    "image_path": str(img_root.resolve()),
                    "parquet_path": str(ingest_out.resolve()),
                    "weight": 1.0,
                }
            ],
            "columns": ["id", "image", "conversations", "image_path"],
            "export": {
                "output_path": "./output",
                "format": "parquet",
                "include_absolute_paths": True,
            },
            "seed": 42,
        }
        mode = "colon_gpt"

    elif scan_dir is not None:
        try:
            config_dict = YAMLConfigLoader.create_config_from_scan(scan_dir)
            mode = "scanned"
        except Exception as e:
            rprint(f"❌ Failed to scan directory {scan_dir}: {e}")
            raise typer.Exit(1)
    else:
        config_dict = YAMLConfigLoader.create_example_config()
        mode = "example"

    with open(output, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    if mode == "scanned":
        rprint(f"✅ Scanned configuration created at: {output}")
        rprint("ℹ️  Edit the file to adjust weights, columns, and task proportions as needed.")
    elif mode == "colon_gpt":
        rprint(f"✅ ColonGPT configuration created at: {output}")
        rprint(f"📁 Fake data scaffolded under: {(_Path(__file__).resolve().parents[2] / 'tests' / 'data')}")
        rprint("💡 Next: run 'endofactory ingest config.yaml' or 'endofactory build config.yaml' to generate parquet and mix.")
    else:
        rprint(f"✅ Example configuration created at: {output}")


def _scaffold_colon_gpt_fake(td_root: Path, json_root: Path, img_root: Path) -> None:
    """Create fake ColonGPT-style JSON and image files for SUN and LUNA datasets.

    Structure:
    - Json-file/{train,test}/*.json
    - Positive-images/{SUN,LUNA}/{Train,Test}/...jpg
    """
    import json as _json
    td_root.mkdir(parents=True, exist_ok=True)
    for d in [json_root / 'train', json_root / 'test']:
        d.mkdir(parents=True, exist_ok=True)

    # Create minimal fake images
    def _touch(p: Path):
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_bytes(b"")

    # Two datasets: SUN and LUNA
    for ds in ['SUN', 'LUNA']:
        for split_name in ['Train', 'Test']:
            # create two images per split
            for i in [1, 2]:
                fname = f"{ds}_{split_name.lower()}_{i:04d}.jpg"
                rel = f"{ds}/{split_name}/class_x/{fname}"
                _touch(img_root / rel)

    # Build JSON entries
    def make_item(ds: str, split_cap: str, idx: int) -> dict:
        file_id = f"{ds}/{split_cap}/high_grade_adenoma/{ds}_{split_cap.lower()}_{idx:04d}.jpg"
        return {
            "id": file_id,
            "image": f"{ds}_{split_cap.lower()}_{idx:04d}",
            "conversations": [
                {"from": "human", "value": "<image>\nCategorize the object."},
                {"from": "gpt", "value": "high grade adenoma"}
            ]
        }

    train_items = [
        make_item('SUN', 'Train', 1),
        make_item('SUN', 'Train', 2),
        make_item('LUNA', 'Train', 1),
        make_item('LUNA', 'Train', 2),
    ]
    test_items = [
        make_item('SUN', 'Test', 1),
        make_item('SUN', 'Test', 2),
        make_item('LUNA', 'Test', 1),
        make_item('LUNA', 'Test', 2),
    ]

    with open(json_root / 'train' / 'data.json', 'w', encoding='utf-8') as f:
        _json.dump(train_items, f, ensure_ascii=False, indent=2)
    with open(json_root / 'test' / 'data.json', 'w', encoding='utf-8') as f:
        _json.dump(test_items, f, ensure_ascii=False, indent=2)


def _detect_dataset_prefixes(scan_dir: Path) -> list[str]:
    """Detect dataset prefixes under scan_dir.

    Priority 1: subdirectories under Positive-images/* (folder names are dataset names).
    Fallback: parse Json-file/train|test/data.json ids and take prefix before first '/' if present.
    """
    ds: set[str] = set()
    img_root = scan_dir / 'Positive-images'
    if img_root.exists() and img_root.is_dir():
        for p in img_root.iterdir():
            if p.is_dir():
                ds.add(p.name)
    if ds:
        return sorted(ds)

    # Fallback to JSONs
    import json as _json
    for json_split in ['train', 'test']:
        data_file = scan_dir / 'Json-file' / json_split / 'data.json'
        if data_file.exists():
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = _json.load(f)
                items = data if isinstance(data, list) else [data]
                for it in items:
                    if isinstance(it, dict) and 'id' in it and isinstance(it['id'], str) and '/' in it['id']:
                        ds.add(it['id'].split('/', 1)[0])
            except Exception:
                pass
    return sorted(ds)


@app.command()
def build(
    config: Path = typer.Argument(..., help="Path to YAML configuration file"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress verbose output"),
    auto_ingest: bool = typer.Option(True, help="If input is configured, run ingestion before building")
):
    """Build EndoVQA dataset from configuration."""
    if not config.exists():
        rprint(f"❌ Configuration file not found: {config}")
        raise typer.Exit(1)
    
    try:
        # Load configuration
        verbose = not quiet
        if verbose:
            rprint(f"📖 Loading configuration from {config}")
        
        factory_config = YAMLConfigLoader.load_config(config)
        engine = EndoFactoryEngine(factory_config)

        # Optional ingestion if configured
        if auto_ingest and getattr(factory_config, 'input', None) and getattr(factory_config, 'ingest_output', None):
            if verbose:
                rprint("🧩 Ingestion configured. Scanning JSON and generating parquet...")
            ingested = engine.ingest_from_input()
            if ingested is not None and verbose:
                rprint(f"✅ Ingested {len(ingested)} records to {factory_config.ingest_output.parquet_path}")
        
        # Load datasets
        if verbose:
            rprint("🔄 Loading datasets...")
        engine.load_datasets()
        
        # Show dataset statistics
        stats = engine.get_dataset_stats()
        _display_dataset_stats(stats, is_colon_gpt=_is_colon_gpt_config(factory_config))
        
        # Mix datasets
        if verbose:
            rprint("🔀 Mixing datasets...")
        mixed_df = engine.mix_datasets()

        if mixed_df is None or len(mixed_df) == 0:
            rprint("ℹ️ No samples available to mix or export. Skipping file generation.")
            return
        
        # Export dataset
        if verbose:
            rprint("💾 Exporting dataset...")
        output_file = engine.export_dataset()
        
        rprint(f"✅ Dataset successfully created at: {output_file}")
        rprint(f"📊 Total samples: {len(mixed_df)}")
        
    except Exception as e:
        rprint(f"❌ Error: {e}")
        raise typer.Exit(1)


@app.command()
def stats(
    config: Path = typer.Argument(..., help="Path to YAML configuration file")
):
    """Show statistics for configured datasets."""
    if not config.exists():
        rprint(f"❌ Configuration file not found: {config}")
        raise typer.Exit(1)
    
    try:
        factory_config = YAMLConfigLoader.load_config(config)
        engine = EndoFactoryEngine(factory_config)
        engine.load_datasets()
        
        stats = engine.get_dataset_stats()
        _display_dataset_stats(stats)
        
    except Exception as e:
        rprint(f"❌ Error: {e}")
        raise typer.Exit(1)


@app.command()
def ingest(
    config: Path = typer.Argument(..., help="Path to YAML configuration file"),
    verbose: bool = typer.Option(True, "--verbose", "-v", help="Verbose output"),
):
    """Run input ingestion (e.g., ColonGPT JSON -> parquet) as configured in YAML.

    Requirements in config:
    - input: { inputset: ColonGPT, json_dir: ..., image_path_mode: 'join_id'|'use_existing', images_root?: ... }
    - ingest_output: { parquet_path: ..., dataset_name?: ... }
    """
    if not config.exists():
        rprint(f"❌ Configuration file not found: {config}")
        raise typer.Exit(1)

    try:
        factory_config = YAMLConfigLoader.load_config(config)
        if not getattr(factory_config, 'input', None) or not getattr(factory_config, 'ingest_output', None):
            rprint("ℹ️  No ingestion configuration found in YAML (input/ingest_output). Nothing to do.")
            raise typer.Exit(0)

        engine = EndoFactoryEngine(factory_config)
        if verbose:
            rprint("🧩 Starting ingestion...")
        df = engine.ingest_from_input()
        if df is None:
            rprint("ℹ️  Ingestion skipped (unsupported inputset or empty).")
            raise typer.Exit(0)
        rprint(f"✅ Ingested {len(df)} records -> {factory_config.ingest_output.parquet_path}")
    except Exception as e:
        rprint(f"❌ Error: {e}")
        raise typer.Exit(1)


@app.command()
def view(
    parquet_file: Path = typer.Argument(..., help="Path to parquet file to visualize"),
    rows: int = typer.Option(5, "--rows", "-n", help="Number of rows to display"),
    columns: bool = typer.Option(True, "--columns", help="Show column information")
):
    """Visualize parquet file structure and sample data."""
    if not parquet_file.exists():
        rprint(f"❌ Parquet file not found: {parquet_file}")
        raise typer.Exit(1)
    
    try:
        import polars as pl
        
        # Load parquet file
        df = pl.read_parquet(parquet_file)
        
        # Display file info
        rprint(f"📁 File: {parquet_file}")
        rprint(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        rprint(f"💾 Size: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Display column information
        if columns:
            rprint("\n📋 **Column Information:**")
            column_table = Table(title="Columns")
            column_table.add_column("Column", style="cyan")
            column_table.add_column("Type", style="magenta")
            column_table.add_column("Non-null Count", style="green")
            column_table.add_column("Sample Values", style="yellow")
            
            for col in df.columns:
                col_type = str(df[col].dtype)
                non_null_count = df[col].count()
                
                # Get sample values (handle different data types)
                try:
                    if df[col].dtype == pl.List:
                        # For list columns, show first few non-null values
                        sample_values = df[col].drop_nulls().limit(3).to_list()
                        sample_str = ", ".join([str(v)[:30] + "..." if len(str(v)) > 30 else str(v) for v in sample_values])
                    else:
                        # For other columns, get unique values
                        sample_values = df[col].drop_nulls().unique().limit(3).to_list()
                        sample_str = ", ".join([str(v)[:20] + "..." if len(str(v)) > 20 else str(v) for v in sample_values])
                        if len(df[col].unique()) > 3:
                            sample_str += "..."
                    
                    if len(sample_values) == 0:
                        sample_str = "All null"
                except Exception:
                    # Fallback for any problematic columns
                    sample_str = "Complex data type"
                
                column_table.add_row(
                    col,
                    col_type,
                    str(non_null_count),
                    sample_str
                )
            
            console.print(column_table)
        
        # Display sample data
        rprint(f"\n🔍 **First {rows} rows:**")
        sample_df = df.head(rows)
        
        # Create a display table
        data_table = Table(title=f"Sample Data (First {rows} rows)")
        
        # Add columns to table
        for col in df.columns:
            data_table.add_column(col, style="white", max_width=30)
        
        # Add rows to table
        for row in sample_df.iter_rows():
            formatted_row = []
            for value in row:
                if value is None:
                    formatted_row.append("[dim]null[/dim]")
                elif isinstance(value, str) and len(value) > 30:
                    formatted_row.append(value[:27] + "...")
                elif isinstance(value, list):
                    list_str = str(value)
                    if len(list_str) > 30:
                        formatted_row.append(list_str[:27] + "...")
                    else:
                        formatted_row.append(list_str)
                else:
                    formatted_row.append(str(value))
            data_table.add_row(*formatted_row)
        
        console.print(data_table)
        
        # Display value counts for categorical columns
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype in [pl.Utf8, pl.Categorical] and df[col].n_unique() <= 20:
                categorical_cols.append(col)
        
        if categorical_cols:
            rprint(f"\n📈 **Value Counts for Categorical Columns:**")
            for col in categorical_cols[:3]:  # Show first 3 categorical columns
                value_counts = df[col].value_counts().sort("count", descending=True).head(10)
                rprint(f"\n**{col}:**")
                for row in value_counts.iter_rows(named=True):
                    rprint(f"  • {row[col]}: {row['count']}")
        
    except Exception as e:
        rprint(f"❌ Error reading parquet file: {e}")
        raise typer.Exit(1)


def _is_colon_gpt_config(config) -> bool:
    """Check if this is a ColonGPT configuration."""
    # Check if any dataset has json_dir (indicates ColonGPT direct ingestion)
    # or if there's an input.inputset == 'ColonGPT'
    if hasattr(config, 'input') and config.input and getattr(config.input, 'inputset', None) == 'ColonGPT':
        return True
    
    if hasattr(config, 'datasets') and config.datasets:
        for dataset in config.datasets:
            if hasattr(dataset, 'json_dir') and dataset.json_dir:
                return True
    
    return False


def _display_dataset_stats(stats: dict, is_colon_gpt: bool = False):
    """Display dataset statistics in a formatted table."""
    if is_colon_gpt:
        # Simplified table for ColonGPT - no Task/Subtask columns
        table = Table(title="Dataset Statistics")
        table.add_column("Dataset", style="cyan")
        table.add_column("Samples", style="magenta")
        
        for name, stat in stats.items():
            if name != "mixed_dataset":
                table.add_row(
                    name,
                    str(stat.get("total_samples", 0))
                )
    else:
        # Full table for other dataset types
        table = Table(title="Dataset Statistics")
        table.add_column("Dataset", style="cyan")
        table.add_column("Samples", style="magenta")
        table.add_column("Tasks", style="green")
        table.add_column("Subtasks", style="yellow")
        
        for name, stat in stats.items():
            if name != "mixed_dataset":
                tasks = ", ".join(stat.get("tasks", [])[:3])  # Show first 3 tasks
                if len(stat.get("tasks", [])) > 3:
                    tasks += "..."
                
                subtasks = ", ".join(stat.get("subtasks", [])[:3])  # Show first 3 subtasks
                if len(stat.get("subtasks", [])) > 3:
                    subtasks += "..."
                
                table.add_row(
                    name,
                    str(stat["total_samples"]),
                    tasks or "N/A",
                    subtasks or "N/A"
                )
    
    console.print(table)
    
    if "mixed_dataset" in stats:
        rprint(f"\n🎯 Mixed Dataset: {stats['mixed_dataset']['total_samples']} samples")
        if stats['mixed_dataset']['source_distribution']:
            rprint("📊 Source Distribution:")
            for source, count in stats['mixed_dataset']['source_distribution'].items():
                rprint(f"  • {source}: {count} samples")


@app.command()
def clear_cache(
    scan_dir: Path = typer.Argument(..., help="ColonGPT base directory containing Json-file/ and Positive-images/"),
):
    """Clear ColonGPT union parquet cache(s) under Json-file/*/.endofactory_cache.

    It removes the directory `.endofactory_cache` in both Json-file/train and Json-file/test if present.
    """
    removed = []
    for split in ["train", "test"]:
        cache_dir = scan_dir / "Json-file" / split / ".endofactory_cache"
        if cache_dir.exists() and cache_dir.is_dir():
            try:
                shutil.rmtree(cache_dir)
                removed.append(str(cache_dir))
            except Exception as e:
                rprint(f"❌ Failed to remove cache dir {cache_dir}: {e}")
    if removed:
        rprint("🧹 Removed caches:\n" + "\n".join(f"  • {p}" for p in removed))
    else:
        rprint("ℹ️  No cache directories found to remove.")


if __name__ == "__main__":
    app()
