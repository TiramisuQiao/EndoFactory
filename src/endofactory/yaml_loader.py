"""YAML configuration loader for EndoFactory."""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from .config import EndoFactoryConfig


class YAMLConfigLoader:
    """Loader for YAML configuration files."""
    
    @staticmethod
    def load_config(config_path: Path) -> EndoFactoryConfig:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return EndoFactoryConfig(**config_dict)
    
    @staticmethod
    def save_config(config: EndoFactoryConfig, output_path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = config.model_dump()
        
        # Convert Path objects to strings for YAML serialization
        YAMLConfigLoader._convert_paths_to_strings(config_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def _convert_paths_to_strings(obj: Any) -> None:
        """Recursively convert Path objects to strings."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, Path):
                    obj[key] = str(value)
                elif isinstance(value, (dict, list)):
                    YAMLConfigLoader._convert_paths_to_strings(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, Path):
                    obj[i] = str(item)
                elif isinstance(item, (dict, list)):
                    YAMLConfigLoader._convert_paths_to_strings(item)
    
    @staticmethod
    def create_example_config() -> Dict[str, Any]:
        """Create an example configuration dictionary."""
        # Match example_config.yaml structure and paths
        repo_root = Path(__file__).resolve().parents[2]
        td = repo_root / "test_data"

        v1_img_str = str(td / "endoscopy_vqa_v1" / "images")
        v1_parquet_str = str(td / "endoscopy_vqa_v1" / "metadata.parquet")
        v2_img_str = str(td / "medical_vqa_v2" / "images")
        v2_parquet_str = str(td / "medical_vqa_v2" / "metadata.parquet")

        return {
            "datasets": [
                {
                    "name": "endoscopy_vqa_v1",
                    "image_path": v1_img_str,
                    "parquet_path": v1_parquet_str,
                    "weight": 0.6,
                },
                {
                    "name": "medical_vqa_v2",
                    "image_path": v2_img_str,
                    "parquet_path": v2_parquet_str,
                    "weight": 0.4,
                }
            ],
            "columns": [
                "uuid",
                "question",
                "answer",
                "options",
                "task",
                "subtask",
                "category",
                "scene",
            ],
            "task_proportions": {
                "task_proportions": {
                    "classification": 0.4,
                    "detection": 0.3,
                    "segmentation": 0.3
                },
                "subtask_proportions": {
                    "classification": {
                        "organ_classification": 0.5,
                        "disease_classification": 0.5
                    },
                    "detection": {
                        "polyp_detection": 0.7,
                        "lesion_detection": 0.3
                    }
                }
            },
            "export": {
                "output_path": "./output",
                "format": "parquet",
                "include_absolute_paths": True
            },
            "seed": 42
        }

    @staticmethod
    def create_config_from_scan(scan_dir: Path, *, default_output: Path = Path("./output")) -> Dict[str, Any]:
        """Create a configuration by scanning ONLY one level deep.

        Supported topologies (non-recursive):
        1) scan_dir has multiple dataset subfolders (e.g., test_data/dataset_a/, dataset_b/),
           and each subfolder directly contains a .parquet and an image directory.
        2) scan_dir itself directly contains a .parquet and an image directory.

        Rules:
        - Do not recurse beyond one level.
        - Image directories are those whose name contains "image" (e.g., images, image, imgs...).
        - Dataset name defaults to subfolder name in case (1); if multiple parquets in a subfolder,
          name will use the parquet stem.
        - Only include pairs that have BOTH image_path and parquet_path.
        - weight defaults to 1.0.
        """
        scan_dir = Path(scan_dir)
        if not scan_dir.exists() or not scan_dir.is_dir():
            raise FileNotFoundError(f"Scan directory not found or not a directory: {scan_dir}")
        
        # Helpers to list immediate parquets and image dirs inside a directory
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        def list_top_parquets(d: Path) -> List[Path]:
            return [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".parquet"]

        def dir_has_top_images(d: Path) -> bool:
            try:
                for p in d.iterdir():
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                        return True
            except Exception:
                return False
            return False

        def list_top_image_dirs(d: Path) -> List[Path]:
            dirs: List[Path] = []
            for p in d.iterdir():
                if not p.is_dir():
                    continue
                if "image" in p.name.lower() or dir_has_top_images(p):
                    dirs.append(p)
            return dirs

        datasets: List[Dict[str, Any]] = []

        # Case (1): scan_dir has dataset subfolders
        subdirs = [p for p in scan_dir.iterdir() if p.is_dir()]
        if subdirs:
            for sd in subdirs:
                parquets = list_top_parquets(sd)
                img_dirs = list_top_image_dirs(sd)
                # If the subdir itself contains images directly, treat it as an image dir
                if not img_dirs and dir_has_top_images(sd):
                    img_dirs = [sd]
                if not img_dirs or not parquets:
                    continue
                # If single parquet and single image dir, name by subdir
                if len(parquets) == 1 and len(img_dirs) == 1:
                    datasets.append(
                        {
                            "name": sd.name,
                            "image_path": str(img_dirs[0]),
                            "parquet_path": str(parquets[0]),
                            "weight": 1.0,
                        }
                    )
                else:
                    # Multiple parquets or images: try name-based pairing
                    used_images: set[Path] = set()
                    for pq in parquets:
                        match: Optional[Path] = None
                        for img in img_dirs:
                            if img in used_images:
                                continue
                            if pq.stem.lower() in img.name.lower() or img.name.lower() in pq.stem.lower():
                                match = img
                                break
                        if match is not None:
                            datasets.append(
                                {
                                    "name": pq.stem,
                                    "image_path": str(match),
                                    "parquet_path": str(pq),
                                    "weight": 1.0,
                                }
                            )
                            used_images.add(match)

        # Case (1.b): root-level parquets matched with images inside subdirectories with matching names
        # Example: scan_dir/endoscopy_vqa_v1.parquet pairs with scan_dir/endoscopy_vqa_v1/images
        root_parquets = list_top_parquets(scan_dir)
        if subdirs and root_parquets:
            for pq in root_parquets:
                matched_sd = None
                for sd in subdirs:
                    if pq.stem.lower() in sd.name.lower() or sd.name.lower() in pq.stem.lower():
                        matched_sd = sd
                        break
                if matched_sd is not None:
                    sd_imgs = list_top_image_dirs(matched_sd)
                    if not sd_imgs and dir_has_top_images(matched_sd):
                        sd_imgs = [matched_sd]
                    chosen_img: Optional[Path] = None
                    if len(sd_imgs) == 1:
                        chosen_img = sd_imgs[0]
                    else:
                        for img in sd_imgs:
                            if pq.stem.lower() in img.name.lower() or img.name.lower() in pq.stem.lower():
                                chosen_img = img
                                break
                    if chosen_img is not None:
                        datasets.append(
                            {
                                "name": pq.stem,
                                "image_path": str(chosen_img),
                                "parquet_path": str(pq),
                                "weight": 1.0,
                            }
                        )

        # Case (2): scan_dir itself contains files/dirs to pair
        if not datasets:
            parquets = list_top_parquets(scan_dir)
            img_dirs = list_top_image_dirs(scan_dir)
            if len(parquets) == 1 and len(img_dirs) == 1:
                datasets.append(
                    {
                        "name": scan_dir.name,
                        "image_path": str(img_dirs[0]),
                        "parquet_path": str(parquets[0]),
                        "weight": 1.0,
                    }
                )
            else:
                used_images: set[Path] = set()
                for pq in parquets:
                    match: Optional[Path] = None
                    for img in img_dirs:
                        if img in used_images:
                            continue
                        if pq.stem.lower() in img.name.lower() or img.name.lower() in pq.stem.lower():
                            match = img
                            break
                    if match is not None:
                        datasets.append(
                            {
                                "name": pq.stem,
                                "image_path": str(match),
                                "parquet_path": str(pq),
                                "weight": 1.0,
                            }
                        )
                        used_images.add(match)

        if not datasets:
            raise ValueError(
                f"No dataset pairs found under {scan_dir}. Place .parquet files and image directories at the same level."
            )

        config: Dict[str, Any] = {
            "datasets": datasets,
            # Keep columns optional; users can add later
            "columns": [
                "uuid",
                "question",
                "answer",
                "options",
                "task",
                "subtask",
                "category",
                "scene",
            ],
            # task_proportions left minimal/empty so pydantic allows it
            "task_proportions": {
                "task_proportions": {},
                "subtask_proportions": {},
            },
            "export": {
                "output_path": str(default_output),
                "format": "parquet",
                "include_absolute_paths": True,
            },
            "seed": 42,
        }

        return config
