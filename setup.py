from pathlib import Path
import re
from setuptools import setup, find_packages

BASE_DIR = Path(__file__).parent
README = (BASE_DIR / "README.md").read_text(encoding="utf-8")
VERSION_FILE = BASE_DIR / "src" / "endofactory" / "__init__.py"


def read_version():
    content = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", content, re.M)
    if not match:
        raise RuntimeError("Cannot find __version__ in src/endofactory/__init__.py")
    return match.group(1)


setup(
    name="endofactory",
    version=read_version(),
    description="Revolutionary EndoVQA dataset construction tool for rapid dataset mixing and configuration",
    long_description=README,
    long_description_content_type="text/markdown",
    author="TiramisuQiao",
    author_email="tlmsq@outlook.com",
    url="https://github.com/TiramisuQiao/EndoFactory",
    project_urls={
        "Repository": "https://github.com/TiramisuQiao/EndoFactory",
        "Issues": "https://github.com/TiramisuQiao/EndoFactoryissues",
    },
    license="MIT",
    keywords=["endoscopy", "medical", "VQA", "dataset", "polars", "yaml", "cli"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "polars>=0.20.0",
        "pyyaml>=6.0",
        "pillow>=10.0.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "pydantic>=2.0.0",
        "uuid>=1.30",
    ],
    entry_points={
        "console_scripts": [
            "endofactory=endofactory.cli:app",
        ]
    },
)
