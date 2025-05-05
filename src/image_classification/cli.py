import argparse

import torchvision.models as models
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from image_classification.conf import load_config

# Get available model architectures from torchvision
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
model_names_str = " | ".join(model_names)


def parse_args() -> argparse.Namespace:
    """Process CLI arguments without setting defaults to avoid overriding YAML config."""
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    # Config file - required
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")

    # Dataset path
    parser.add_argument("--data-dir", metavar="DIR", help="Path to dataset (overrides config)")

    # Model architecture
    parser.add_argument(
        "-a",
        "--model-arch",
        metavar="ARCH",
        choices=model_names,
        help=f"Model architecture (overrides config), options: {model_names_str}",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=None,
        help="Use pre-trained model (overrides config)",
    )

    # Workers
    parser.add_argument(
        "-j",
        "--data-workers",
        type=int,
        metavar="N",
        help="Number of data loading workers (overrides config)",
    )

    # Epochs
    parser.add_argument(
        "--epochs", type=int, metavar="N", help="Number of total epochs to run (overrides config)"
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        metavar="N",
        help="Manual epoch number for restarts (overrides config)",
    )

    # Hyperparams
    parser.add_argument(
        "-b", "--batch-size", type=int, metavar="N", help="Mini-batch size (overrides config)"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        metavar="LR",
        help="Initial learning rate (overrides config)",
        dest="lr",
    )
    parser.add_argument("--momentum", type=float, metavar="M", help="Momentum (overrides config)")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        type=float,
        metavar="W",
        help="Weight decay (overrides config)",
        dest="weight_decay",
    )

    # Output and checkpointing
    parser.add_argument(
        "-p", "--print-freq", type=int, metavar="N", help="Print frequency (overrides config)"
    )
    parser.add_argument(
        "--resume", type=str, metavar="PATH", help="Path to latest checkpoint (overrides config)"
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        action="store_true",
        default=None,
        help="Evaluate model on validation set (overrides config)",
    )

    # Distributed training
    parser.add_argument(
        "--world-size", type=int, help="Number of nodes for distributed training (overrides config)"
    )
    parser.add_argument(
        "--rank", type=int, help="Node rank for distributed training (overrides config)"
    )
    parser.add_argument(
        "--dist-url", type=str, help="URL used to set up distributed training (overrides config)"
    )
    parser.add_argument("--dist-backend", type=str, help="Distributed backend (overrides config)")
    parser.add_argument(
        "--seed", type=int, help="Seed for initializing training (overrides config)"
    )
    parser.add_argument("--gpu", type=int, help="GPU ID to use (overrides config)")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        default=None,
        help="Use multi-processing distributed training (overrides config)",
    )

    # Dummy data
    parser.add_argument(
        "--dummy",
        action="store_true",
        default=None,
        help="Use fake data to benchmark (overrides config)",
    )

    args = parser.parse_args()
    return args


def parse_args_and_config() -> tuple[argparse.Namespace, DictConfig]:
    """Parse command-line arguments and load configuration.

    Returns:
        Tuple of parsed arguments and configuration
    """
    args = parse_args()

    # Load base config from YAML
    config = load_config(args.config)

    # Create a list of override values from non-None arguments
    overrides = []
    for arg_name, arg_value in vars(args).items():
        # Skip the config file argument and None values
        if arg_name != "config" and arg_value is not None:
            # Convert arg_name from snake_case to dotted format for nested configs
            if arg_name in ["data_dir", "data_workers", "batch_size", "use_dummy_data"]:
                config_key = f"data.{arg_name}"
            elif arg_name in ["model_arch", "pretrained"]:
                config_key = f"model.{arg_name}"
            elif arg_name in [
                "epochs",
                "start_epoch",
                "lr",
                "momentum",
                "weight_decay",
                "print_freq",
                "resume",
                "evaluate",
            ]:
                config_key = f"training.{arg_name}"
            elif arg_name in [
                "world_size",
                "rank",
                "dist_url",
                "dist_backend",
                "multiprocessing_distributed",
                "gpu",
                "seed",
            ]:
                config_key = f"distributed.{arg_name}"
            else:
                # Unknown argument
                continue

            # Add to override list
            overrides.append(f"{config_key}={arg_value}")

    # Apply overrides if any exist
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        config: DictConfig = OmegaConf.merge(config, cli_conf)  # type: ignore

    logger.info(f"Loaded configuration: \n{OmegaConf.to_yaml(config)}")

    return args, config


if __name__ == "__main__":
    # Simple test to see if the CLI parser works
    args, config = parse_args_and_config()
