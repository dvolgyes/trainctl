"""CIFAR-10 CNN classifier exercising TrainctlMixin against a real Lightning run."""

from dataclasses import dataclass

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn, optim
from torchmetrics.classification import MulticlassAccuracy
from trainctl import TrainctlMixin

NUM_CLASSES = 10


@dataclass(frozen=True)
class OptimHParams:
    """Optimizer and learning-rate-schedule hyperparameters for `CifarClassifier`.

    Attributes:
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay (L2 regularization).
        lr_step_size: Epoch interval between learning-rate decay steps.
        lr_gamma: Multiplicative learning-rate decay factor.
    """

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_step_size: int = 30
    lr_gamma: float = 0.1


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Builds one conv-batchnorm-relu-maxpool block that halves spatial resolution.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.

    Returns:
        The assembled block.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


def _build_backbone() -> nn.Sequential:
    """Builds a compact convolutional feature extractor for 32x32 RGB images.

    Returns:
        Sequential module mapping (N, 3, 32, 32) inputs to (N, 128, 1, 1) feature maps.
    """
    return nn.Sequential(
        _conv_block(3, 32),
        _conv_block(32, 64),
        _conv_block(64, 128),
        nn.AdaptiveAvgPool2d(output_size=1),
    )


class CifarClassifier(TrainctlMixin, pl.LightningModule):
    """Compact CNN classifier for CIFAR-10, composed with TrainctlMixin under test.

    Args:
        hp: Optimizer and learning-rate-schedule hyperparameters, logged to the
            configured logger's hyperparameter tracking (e.g. TensorBoard's HPARAMS tab).
        **trainctl_kwargs: Forwarded to `TrainctlMixin.__init__` (e.g. `rest_port`,
            `fuse_mountpoint`, `fuse_enabled`).

    Attributes:
        hp: The optimizer and learning-rate-schedule hyperparameters in effect.
        backbone: Convolutional feature extractor.
        head: Linear classification head.
        train_accuracy: Running multiclass accuracy metric for training batches.
        val_accuracy: Running multiclass accuracy metric for validation batches.
    """

    hp: OptimHParams
    backbone: nn.Sequential
    head: nn.Linear
    train_accuracy: MulticlassAccuracy
    val_accuracy: MulticlassAccuracy

    # Args are documented on the class docstring per pydoclint's DOC301, which forbids a
    # separate __init__ docstring here (see README: known ruff-D107/pydoclint-DOC301 conflict).
    def __init__(
        self, hp: OptimHParams | None = None, **trainctl_kwargs: object
    ) -> None:
        super().__init__(**trainctl_kwargs)
        self.hp = hp or OptimHParams()
        self.save_hyperparameters(vars(self.hp))
        self.backbone = _build_backbone()
        self.head = nn.Linear(128, NUM_CLASSES)
        self.train_accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES)
        self.val_accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES)

    # Lightning replaces the base `*args, **kwargs` signature with the module's own
    # concrete inputs by design; pylint's arguments-differ check does not model that.
    def forward(self, images: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        """Computes class logits for a batch of images.

        Args:
            images: Batch of shape (N, 3, 32, 32).

        Returns:
            Class logits of shape (N, NUM_CLASSES).
        """
        features = self.backbone(images).flatten(start_dim=1)
        logits: Tensor = self.head(features)
        return logits

    def _forward_batch(
        self, batch: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Runs a forward pass and computes the cross-entropy loss for one batch.

        Args:
            batch: Pair of (images, labels) tensors.

        Returns:
            The loss, the predicted logits, and the ground-truth labels.
        """
        images, labels = batch
        logits = self(images)
        loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits, labels

    def training_step(  # pylint: disable=arguments-differ
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Computes and logs the training loss and running accuracy for one batch.

        Args:
            batch: Pair of (images, labels) tensors.
            batch_idx: Index of the batch within the current epoch; unused.

        Returns:
            The training loss to backpropagate.
        """
        del batch_idx
        loss, logits, labels = self._forward_batch(batch)
        self.train_accuracy(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Computes and logs the validation loss and running accuracy for one batch.

        Args:
            batch: Pair of (images, labels) tensors.
            batch_idx: Index of the batch within the current epoch; unused.

        Returns:
            The validation loss.
        """
        del batch_idx
        loss, logits, labels = self._forward_batch(batch)
        self.val_accuracy(logits, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Builds the Adam optimizer and step learning-rate scheduler.

        Returns:
            The optimizer and learning-rate scheduler configuration.
        """
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hp.learning_rate,
            weight_decay=self.hp.weight_decay,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hp.lr_step_size, gamma=self.hp.lr_gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
