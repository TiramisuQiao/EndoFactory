# EndoFactory 🏭

![EndoFactory Logo](assets/logo.png)

Revolutionary tool for constructing EndoVQA datasets through YAML configuration.

[![codecov](https://codecov.io/github/TiramisuQiao/EndoFactory/graph/badge.svg?token=N4SZ3BLO4P)](https://codecov.io/github/TiramisuQiao/EndoFactory)

## Quick Start

### 1. Installation

If you just want the cli, pip it!

```bash
pip install endofactory -i https://pypi.org/simple
```

Or if you want to be contributor,

```bash
git clone <repository-url>
cd EndoFactory
poetry install
```

### 2. Generate Test Data

```bash
poetry run python tests/test_data_generator.py
```

### 3. Create Configuration

```bash
poetry run python -m endofactory.cli create-config --output config.yaml
```

### 4. Edit Configuration

```yaml
datasets:
  - name: endoscopy_vqa_v1
    image_path: /path/to/images
    parquet_path: /path/to/metadata.parquet
    weight: 0.6
  - name: medical_vqa_v2
    image_path: /path/to/images2
    parquet_path: /path/to/metadata2.parquet
    weight: 0.4

columns:
  - uuid
  - question
  - answer
  - options
  - task
  - category

task_proportions:
  task_proportions:
    classification: 0.5
    detection: 0.3
    segmentation: 0.2

export:
  output_path: /path/to/output
  format: parquet
```

### 5. Build Dataset

```bash
poetry run python -m endofactory.cli build config.yaml --verbose
```

### 6. View Results

```bash
poetry run python -m endofactory.cli view output/endovqa_dataset.parquet
```

## CLI Commands

### `create-config`

Generate example configuration file

```bash
endofactory create-config [--output CONFIG_PATH]
```

### `build`

Build mixed dataset from configuration

```bash
endofactory build CONFIG_PATH [--verbose]
```

### `stats`

Show dataset statistics

```bash
endofactory stats CONFIG_PATH
```

### `view`

Visualize parquet file structure and data

```bash
endofactory view PARQUET_FILE [--rows N] [--columns]
```

## Configuration Options

### Dataset Weights

Control proportion of each dataset in final mix:

```yaml
datasets:
  - name: dataset_a
    weight: 0.7  # 70% from dataset_a
  - name: dataset_b  
    weight: 0.3  # 30% from dataset_b
```

### Task Proportions

Control distribution of different task types:

```yaml
task_proportions:
  task_proportions:
    classification: 0.4
    detection: 0.4
    segmentation: 0.2
  subtask_proportions:
    classification:
      organ_classification: 0.6
      disease_classification: 0.4
```

### Global Columns

Specify columns to extract (missing columns filled with null):

```yaml
columns:
  - uuid
  - question
  - answer
  - task
```

## Features

- **🚀 Fast Dataset Mixing**: YAML-based configuration for dataset blending
- **📊 Task Proportion Control**: Precise control over task/subtask distribution
- **💾 Multiple Export Formats**: Support for Parquet and JSONL output
- **🔍 Data Visualization**: One-command parquet file inspection
- **🛡️ Schema Flexibility**: Automatic handling of different column structures
- **⚡ High Performance**: Polars-powered data processing
- **🎯 Reproducible**: Configurable random seeds

## Project Structure

```bash
EndoFactory/
├── src/endofactory/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── core.py
│   └── yaml_loader.py
├── tests/
├── assets/
├── example_config.yaml
└── pyproject.toml
```

## Requirements

- Python >= 3.9
- Poetry (recommended) or pip

## License

MIT License
