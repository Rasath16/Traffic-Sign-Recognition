"""Configuration settings for the Traffic Sign Recognition project."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    notebooks_dir: Path = project_root / "notebooks"
    results_dir: Path = project_root / "results"


cfg = Config()
