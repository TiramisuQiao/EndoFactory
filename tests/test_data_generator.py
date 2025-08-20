"""Generate test datasets with fake images and VQA parquet files."""

import uuid
import polars as pl
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
from typing import List, Dict


class TestDataGenerator:
    """Generator for test EndoVQA datasets."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def generate_fake_image(self, image_id: str, width: int = 224, height: int = 224) -> Path:
        """Generate a fake medical image with random colors and text."""
        # Create image with random background
        colors = [(255, 182, 193), (173, 216, 230), (144, 238, 144), (255, 218, 185)]
        bg_color = random.choice(colors)
        
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Add some random shapes to simulate medical imagery
        for _ in range(random.randint(3, 8)):
            x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
            x2, y2 = random.randint(width//2, width), random.randint(height//2, height)
            shape_color = tuple(random.randint(0, 255) for _ in range(3))
            
            if random.choice([True, False]):
                draw.ellipse([x1, y1, x2, y2], fill=shape_color)
            else:
                draw.rectangle([x1, y1, x2, y2], fill=shape_color)
        
        # Add image ID text
        try:
            font = ImageFont.load_default()
            draw.text((10, 10), f"ID: {image_id[:8]}", fill=(0, 0, 0), font=font)
        except:
            draw.text((10, 10), f"ID: {image_id[:8]}", fill=(0, 0, 0))
        
        return img
    
    def generate_vqa_data(self, num_samples: int, dataset_name: str) -> List[Dict]:
        """Generate VQA question-answer pairs."""
        
        # Define question templates and answers for different tasks
        templates = {
            "classification": [
                ("What organ is shown in this image?", ["stomach", "colon", "esophagus", "duodenum"]),
                ("What type of tissue is visible?", ["normal", "inflamed", "ulcerated", "polyp"]),
                ("Is this image normal or abnormal?", ["normal", "abnormal"])
            ],
            "detection": [
                ("Are there any polyps visible?", ["yes", "no"]),
                ("Can you detect any lesions?", ["yes", "no"]),
                ("Is there any bleeding visible?", ["yes", "no"])
            ],
            "segmentation": [
                ("What is the main anatomical structure?", ["mucosa", "submucosa", "muscle layer"]),
                ("Identify the region of interest", ["upper left", "center", "lower right"])
            ]
        }
        
        subtasks = {
            "classification": ["organ_classification", "disease_classification", "tissue_classification"],
            "detection": ["polyp_detection", "lesion_detection", "bleeding_detection"],
            "segmentation": ["anatomical_segmentation", "pathology_segmentation"]
        }
        
        categories = ["endoscopy", "colonoscopy", "gastroscopy", "sigmoidoscopy"]
        scenes = ["upper_gi", "lower_gi", "small_bowel", "rectum"]
        
        data = []
        for i in range(num_samples):
            image_uuid = str(uuid.uuid4())
            task = random.choice(list(templates.keys()))
            question_template, possible_answers = random.choice(templates[task])
            answer = random.choice(possible_answers)
            
            # Generate options for multiple choice (some questions)
            if len(possible_answers) > 2:
                options = random.sample(possible_answers, min(4, len(possible_answers)))
                if answer not in options:
                    options[0] = answer
                random.shuffle(options)
            else:
                options = possible_answers
            
            record = {
                "uuid": image_uuid,
                "filename": f"{image_uuid}.jpg",
                "src_path": f"/original/path/{image_uuid}.jpg",
                "dst_path": f"/processed/path/{image_uuid}.jpg", 
                "rel_path": f"{image_uuid}.jpg",
                "stem": image_uuid,
                "ext": ".jpg",
                "found": True,
                "image_path": f"{image_uuid}.jpg",
                "question": question_template,
                "options": options,
                "answer": answer,
                "category": random.choice(categories),
                "dataset": dataset_name,
                "box": None,  # Bounding box coordinates (for detection tasks)
                "gt": answer,  # Ground truth
                "scene": random.choice(scenes),
                "subtask": random.choice(subtasks[task]),
                "task": task,
                "origin_image_path": f"/original/{image_uuid}.jpg"
            }
            data.append(record)
        
        return data
    
    def create_dataset(self, dataset_name: str, num_samples: int = 100) -> Dict[str, Path]:
        """Create a complete test dataset with images and parquet file."""
        dataset_dir = self.base_path / dataset_name
        images_dir = dataset_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate VQA data
        vqa_data = self.generate_vqa_data(num_samples, dataset_name)
        
        # Generate corresponding images
        for record in vqa_data:
            image_id = record["uuid"]
            img = self.generate_fake_image(image_id)
            img_path = images_dir / f"{image_id}.jpg"
            img.save(img_path, "JPEG")
        
        # Create parquet file
        df = pl.DataFrame(vqa_data)
        parquet_path = dataset_dir / "metadata.parquet"
        df.write_parquet(parquet_path)
        
        return {
            "images_dir": images_dir,
            "parquet_path": parquet_path,
            "dataset_dir": dataset_dir
        }


def create_test_datasets():
    """Create multiple test datasets for demonstration."""
    generator = TestDataGenerator(Path("/Users/wangjohan/EndoFactory/test_data"))
    
    # Create two different datasets
    datasets_info = {}
    
    # Dataset 1: Endoscopy VQA v1
    print("Creating endoscopy_vqa_v1 dataset...")
    datasets_info["endoscopy_vqa_v1"] = generator.create_dataset("endoscopy_vqa_v1", 150)
    
    # Dataset 2: Medical VQA v2  
    print("Creating medical_vqa_v2 dataset...")
    datasets_info["medical_vqa_v2"] = generator.create_dataset("medical_vqa_v2", 120)
    
    return datasets_info


if __name__ == "__main__":
    datasets_info = create_test_datasets()
    
    print("\n✅ Test datasets created successfully!")
    for name, paths in datasets_info.items():
        print(f"\n📁 {name}:")
        print(f"  Images: {paths['images_dir']}")
        print(f"  Metadata: {paths['parquet_path']}")
        
        # Show sample data
        df = pl.read_parquet(paths['parquet_path'])
        print(f"  Samples: {len(df)}")
        print(f"  Tasks: {df['task'].unique().to_list()}")
        print(f"  Categories: {df['category'].unique().to_list()}")
