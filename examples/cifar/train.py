"""CLI entry point that trains CifarClassifier for a bounded wall-clock duration.

Run as a module so the package-relative imports resolve, e.g.:

    uv run python -m examples.cifar.train --max-time 00:01:00:00
"""

import sys
from pathlib import Path

import click
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger

from examples.cifar.data import CifarDataModule
from examples.cifar.model import CifarClassifier, OptimHParams

DEFAULT_DATA_DIR = Path("examples/cifar/data")
DEFAULT_LOG_DIR = Path("examples/cifar/lightning_logs")


def _configure_logging(loglevel: str) -> None:
    """Reconfigures Loguru's default sink to the requested level.

    Removes only Loguru's own pre-installed default sink (id 0) rather than every
    currently-registered sink, so this never tears down a run-file sink trainctl
    attaches once training starts.

    Args:
        loglevel: Loguru log level name, e.g. "INFO" or "DEBUG".
    """
    logger.remove(0)
    logger.add(sys.stderr, level=loglevel)


def _build_trainer(
    log_dir: Path, max_time: str, max_epochs: int, accelerator: str
) -> Trainer:
    """Assembles the Lightning Trainer with TensorBoard logging and checkpointing.

    Args:
        log_dir: Root directory for TensorBoard event files and checkpoints.
        max_time: Wall-clock training budget as "DD:HH:MM:SS".
        max_epochs: Epoch cap; -1 leaves the run duration to `max_time`.
        accelerator: Lightning accelerator identifier (e.g. "auto", "gpu", "cpu").

    Returns:
        The configured Trainer instance.
    """
    tb_logger = TensorBoardLogger(save_dir=log_dir, name="cifar")
    checkpoint_callback = ModelCheckpoint(monitor="val/acc", mode="max", save_top_k=1)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        max_time=max_time,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=20,
    )


@click.command()
@click.option(
    "-d",
    "--data-dir",
    type=click.Path(path_type=Path),  # type: ignore[type-var]  # click's Path stub is AnyStr-only
    default=DEFAULT_DATA_DIR,
    show_default=True,
    help="Directory to download/cache CIFAR-10 in.",
)
@click.option(
    "-o",
    "--log-dir",
    type=click.Path(path_type=Path),  # type: ignore[type-var]  # click's Path stub is AnyStr-only
    default=DEFAULT_LOG_DIR,
    show_default=True,
    help="TensorBoard log and checkpoint directory.",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=128,
    show_default=True,
    help="Training batch size.",
)
@click.option(
    "-r",
    "--learning-rate",
    type=float,
    default=1e-3,
    show_default=True,
    help="Adam learning rate.",
)
@click.option(
    "-w",
    "--num-workers",
    type=int,
    default=4,
    show_default=True,
    help="Dataloader worker processes.",
)
@click.option(
    "-t",
    "--max-time",
    type=str,
    default="00:01:00:00",
    show_default=True,
    help="Wall-clock training budget as DD:HH:MM:SS.",
)
@click.option(
    "-e",
    "--max-epochs",
    type=int,
    default=-1,
    show_default=True,
    help="Epoch cap; -1 lets --max-time govern run length.",
)
@click.option(
    "-s",
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
@click.option(
    "-a",
    "--accelerator",
    type=str,
    default="auto",
    show_default=True,
    help="Lightning accelerator (auto, gpu, cpu).",
)
@click.option(
    "-L",
    "--loglevel",
    type=str,
    default="INFO",
    show_default=True,
    help="Loguru log level.",
)
# Click injects every option as a keyword argument at call time; pylint's positional-argument
# count does not account for that.
def main(  # pylint: disable=too-many-positional-arguments
    data_dir: Path,
    log_dir: Path,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    max_time: str,
    max_epochs: int,
    seed: int,
    accelerator: str,
    loglevel: str,
) -> None:
    """Trains CifarClassifier on CIFAR-10 to exercise a LightningModule-compatible base class."""
    _configure_logging(loglevel)
    seed_everything(seed, workers=True)
    logger.info(
        "starting cifar training run: batch_size={} learning_rate={} max_time={} max_epochs={}",
        batch_size,
        learning_rate,
        max_time,
        max_epochs,
    )

    datamodule = CifarDataModule(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
    )
    model = CifarClassifier(hp=OptimHParams(learning_rate=learning_rate))
    trainer = _build_trainer(
        log_dir=log_dir,
        max_time=max_time,
        max_epochs=max_epochs,
        accelerator=accelerator,
    )

    trainer.fit(model=model, datamodule=datamodule)
    logger.info(
        "training finished after {} epochs, {} steps",
        trainer.current_epoch,
        trainer.global_step,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter  # click supplies args from argv at runtime
