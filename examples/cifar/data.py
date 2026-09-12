"""CIFAR-10 LightningDataModule used by the trainctl integration harness."""

from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2 as transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _train_transform() -> transforms.Compose:
    """Builds the training-time augmentation and normalization pipeline.

    Returns:
        A composed transform mapping a PIL image to a normalized tensor.
    """
    return transforms.Compose(
        [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


def _eval_transform() -> transforms.Compose:
    """Builds the evaluation-time normalization pipeline.

    Returns:
        A composed transform mapping a PIL image to a normalized tensor.
    """
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


class CifarDataModule(LightningDataModule):
    """CIFAR-10 datamodule for the trainctl integration harness.

    The torchvision test split is used as the validation split; this harness verifies
    training-loop wiring rather than producing a rigorous benchmark result.

    Args:
        data_dir: Directory to download and cache CIFAR-10 in.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocess data-loading workers.

    Attributes:
        data_dir: Directory CIFAR-10 is downloaded to and read from.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocess data-loading workers.
        train_dataset: CIFAR-10 training split, built in `setup`.
        val_dataset: CIFAR-10 test split used as validation, built in `setup`.
    """

    data_dir: Path
    batch_size: int
    num_workers: int
    train_dataset: CIFAR10
    val_dataset: CIFAR10

    # Args are documented on the class docstring per pydoclint's DOC301, which forbids a
    # separate __init__ docstring here (see README: known ruff-D107/pydoclint-DOC301 conflict).
    def __init__(
        self, data_dir: Path, batch_size: int = 128, num_workers: int = 4
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """Downloads CIFAR-10 into `data_dir` if it is not already present."""
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        """Instantiates the train and validation datasets.

        Args:
            stage: Lightning stage identifier; unused, both splits are always built.
        """
        del stage
        self.train_dataset = CIFAR10(
            root=self.data_dir, train=True, transform=_train_transform()
        )
        self.val_dataset = CIFAR10(
            root=self.data_dir, train=False, transform=_eval_transform()
        )

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Builds the shuffled training dataloader.

        Returns:
            The training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Builds the validation dataloader.

        Returns:
            The validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
