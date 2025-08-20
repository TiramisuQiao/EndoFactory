"""Command-line interface for EndoFactory."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from .yaml_loader import YAMLConfigLoader
from .core import EndoFactoryEngine

app = typer.Typer(help="EndoFactory: Revolutionary EndoVQA dataset construction tool")
console = Console()


@app.command()
def create_config(
    output: Path = typer.Option("config.yaml", help="Output path for configuration file"),
    scan_dir: Optional[Path] = typer.Option(None, help="Optional: scan a directory to auto-detect datasets (images + parquet)")
):
    """Create a configuration file.

    - By default, writes an example configuration.
    - If --scan-dir is provided, auto-detect datasets under the given directory
      by pairing parquet files with nearby images directories.
    """
    import yaml

    if scan_dir is not None:
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
    else:
        rprint(f"✅ Example configuration created at: {output}")


@app.command()
def build(
    config: Path = typer.Argument(..., help="Path to YAML configuration file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Build EndoVQA dataset from configuration."""
    if not config.exists():
        rprint(f"❌ Configuration file not found: {config}")
        raise typer.Exit(1)
    
    try:
        # Load configuration
        if verbose:
            rprint(f"📖 Loading configuration from {config}")
        
        factory_config = YAMLConfigLoader.load_config(config)
        engine = EndoFactoryEngine(factory_config)
        
        # Load datasets
        if verbose:
            rprint("🔄 Loading datasets...")
        engine.load_datasets()
        
        # Show dataset statistics
        stats = engine.get_dataset_stats()
        _display_dataset_stats(stats)
        
        # Mix datasets
        if verbose:
            rprint("🔀 Mixing datasets...")
        mixed_df = engine.mix_datasets()
        
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


def _display_dataset_stats(stats: dict):
    """Display dataset statistics in a formatted table."""
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


if __name__ == "__main__":
    app()
