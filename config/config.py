import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class DataConfig:
    """Data configuration"""
    # Paths
    data_dir: Path = Path("GTSRB")
    train_csv: str = "Train.csv"
    test_csv: str = "Test.csv"
    
    # Image settings
    img_size: Tuple[int, int] = (32, 32)
    img_channels: int = 3
    
    # Split settings
    val_split: float = 0.2
    random_state: int = 42
    
    # Number of classes
    num_classes: int = 43
    
    def __post_init__(self):
        """Validate paths"""
        self.data_dir = Path(self.data_dir)


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    rotation_range: int = 15
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    zoom_range: float = 0.1
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    fill_mode: str = 'nearest'
    horizontal_flip: bool = False  # Traffic signs shouldn't be flipped
    vertical_flip: bool = False


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Common settings
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    num_classes: int = 43
    
    # Custom CNN settings
    filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    dropout_rates: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.4, 0.5])
    dense_units: int = 512
    
    # Transfer learning settings
    base_model_name: str = "MobileNetV2"
    freeze_base: bool = True
    fine_tune_at: int = 100  # Layer to start fine-tuning from


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training parameters
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    
    # Optimizer settings
    optimizer: str = "adam"
    loss: str = "sparse_categorical_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    
    # Callbacks
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_loss"
    
    use_reduce_lr: bool = True
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    reduce_lr_min_lr: float = 1e-7
    
    use_model_checkpoint: bool = True
    checkpoint_monitor: str = "val_accuracy"
    
    # Class balancing
    use_class_weights: bool = True
    
    # Output paths
    model_dir: Path = Path("saved_models")
    log_dir: Path = Path("logs")
    results_dir: Path = Path("results")
    
    def __post_init__(self):
        """Create directories"""
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Visualization settings
    plot_confusion_matrix: bool = True
    plot_training_history: bool = True
    plot_per_class_accuracy: bool = True
    plot_misclassifications: bool = True
    
    # Report settings
    save_classification_report: bool = True
    top_k: int = 5
    
    # Output
    results_dir: Path = Path("results")
    
    def __post_init__(self):
        self.results_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class InferenceConfig:
    """Inference configuration"""
    # Model settings
    model_path: str = "saved_models/custom_cnn_model.h5"
    
    # Grad-CAM settings
    use_gradcam: bool = True
    gradcam_layer: str = None  # Auto-detect if None
    gradcam_alpha: float = 0.4
    
    # Prediction settings
    top_k: int = 5
    confidence_threshold: float = 0.7


@dataclass
class AppConfig:
    """Streamlit app configuration"""
    # App settings
    title: str = "🚦 Traffic Sign Recognition System"
    page_icon: str = "🚦"
    layout: str = "wide"
    
    # Model paths
    models: dict = field(default_factory=lambda: {
        "Custom CNN": "saved_models/custom_cnn_model.h5",
        "MobileNetV2": "saved_models/mobilenet_transfer_model.h5"
    })
    
    # Display settings
    show_gradcam: bool = True
    default_top_k: int = 5
    max_top_k: int = 10


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.data = DataConfig()
        self.augmentation = AugmentationConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.inference = InferenceConfig()
        self.app = AppConfig()
    
    def save(self, filepath: str):
        """Save configuration to file"""
        import json
        from dataclasses import asdict
        
        config_dict = {
            'data': asdict(self.data),
            'augmentation': asdict(self.augmentation),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'evaluation': asdict(self.evaluation),
            'inference': asdict(self.inference),
            'app': asdict(self.app)
        }
        
        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            return obj
        
        config_dict = convert_paths(config_dict)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        # Update config with loaded values
        # (Simplified - in production, properly reconstruct dataclasses)
        
        return config


# Class names mapping
CLASS_NAMES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}


# Create default config instance
default_config = Config()


if __name__ == "__main__":
    # Test configuration
    config = Config()
    print("Configuration loaded successfully!")
    print(f"Data directory: {config.data.data_dir}")
    print(f"Image size: {config.data.img_size}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Number of classes: {config.data.num_classes}")