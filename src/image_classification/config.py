from dataclasses import dataclass, field

import omegaconf
from omegaconf import OmegaConf


@dataclass
class DataConfig:
    data_dir: str = "imagenet"
    workers: int = 4
    batch_size: int = 256
    input_size: tuple[int, int] = (224, 224)
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    use_dummy_data: bool = False


@dataclass
class ModelConfig:
    arch: str = "resnet18"
    pretrained: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 90
    start_epoch: int = 0
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    print_freq: int = 10
    resume: bool = False
    evaluate: bool = False


@dataclass
class DistributedConfig:
    world_size: int = -1
    rank: int = -1
    dist_url: str = "tcp://224.66.41.62:23456"
    dist_backend: str = "nccl"
    multiprocessing_distributed: bool = False
    gpu: int | None = None
    seed: int | None = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


def get_default_config() -> Config:
    """Return the default configuration."""
    return Config()


def load_config(config_path: str | None = None) -> omegaconf.DictConfig:
    """Load configuration from a file or use defaults.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration object
    """
    # Start with structured config defaults
    schema = OmegaConf.structured(Config)

    # Load from file if provided
    config = schema
    if config_path:
        file_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(schema, file_config)

    return config


def save_config(config: omegaconf.DictConfig, path: str) -> None:
    """Save configuration to a file.

    Args:
        config: Configuration object
        path: Path to save the configuration
    """
    OmegaConf.save(config, path)
