"""Deliberately slow live-training run for exercising TrainctlMixin's REST API and FUSE
mountpoint by hand from a second terminal, while training runs in the foreground here.

Run this in one terminal:

    uv run python -m examples.playground.run

Then, in a second terminal, open the REST base URL printed at startup in a
browser (or `curl` it) for a full guide to every endpoint, with links.
Ctrl+C in this terminal stops training gracefully (Lightning's own signal
handling); a second Ctrl+C forces an immediate exit.
"""

import os
import time
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import v2 as T

from trainctl import TrainctlMixin

REST_PORT = 8090
REST_BASE = f"http://127.0.0.1:{REST_PORT}"
LOG_DIR = Path(__file__).resolve().parent / "logs"


def _banner(log_dir: Path) -> str:
    """Builds the usage banner, given this run's resolved TensorBoard log directory.

    Trainctl's FUSE mount and artifact store colocate directly under `log_dir`
    (alongside TensorBoard's own event files) by default -- see
    `trainctl.runtime.runtime._default_base_dir`.
    """
    mnt = log_dir / "procfs"
    artifacts = log_dir / "artifacts"
    return f"""
=================================================================
 trainctl playground -- open {REST_BASE}/ for a full guide + links
                         log dir:    {log_dir}
                         FUSE mount: {mnt}
                         artifacts:  {artifacts}

 Each training step sleeps ~0.3s, so you have time to poke at the
 run from a second terminal. Ctrl+C here stops training gracefully.
=================================================================
"""


class PlaygroundModel(TrainctlMixin, pl.LightningModule):
    """Tiny classifier whose training step sleeps, so a human has time to interact with
    the live run through trainctl's REST API and FUSE mount.

    Attributes:
        net: Small linear classifier over flattened 16x16 RGB inputs.
    """

    trainctl_tunable_hparams = {
        "hparams.step_delay": {"min": 0.0, "max": 5.0, "label": "Step delay (s)"},
    }

    def __init__(self, lr: float = 1e-3, step_delay: float = 0.3) -> None:
        super().__init__(rest_port=REST_PORT)
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(3 * 16 * 16, 32), nn.ReLU(), nn.Linear(32, 10)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Computes class logits for a batch of flattened images.

        Args:
            images: Batch of shape (N, 3, 16, 16).

        Returns:
            Class logits of shape (N, 10).
        """
        logits: torch.Tensor = self.net(images)
        return logits

    def training_step(  # pylint: disable=arguments-differ
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Sleeps briefly, then computes and logs the cross-entropy loss for one batch.

        Args:
            batch: Pair of (images, labels) tensors.
            batch_idx: Index of the batch within the current epoch; unused.

        Returns:
            The training loss to backpropagate.
        """
        del batch_idx
        time.sleep(self.hparams.step_delay)
        images, labels = batch
        loss = nn.functional.cross_entropy(self(images), labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Builds the Adam optimizer.

        Returns:
            The optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def _init_single_process_distributed() -> None:
    """Starts a real, single-rank `torch.distributed` process group.

    `torch.distributed.debug`'s server only activates once a process group is
    initialized (see `trainctl.torch_debug.manager.TorchDebugManager`), which
    normally means a torchrun/DDP launch. A world size of one is enough to
    demonstrate it without a multi-process launch.
    """
    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _teardown_single_process_distributed() -> None:
    """Tears down the process group started by `_init_single_process_distributed`."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _build_loader() -> DataLoader:
    """Builds a small synthetic image-classification dataloader.

    Returns:
        A DataLoader of 16x16 RGB images against 10 fake classes.
    """
    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    dataset = FakeData(
        size=64, image_size=(3, 16, 16), num_classes=10, transform=transform
    )
    return DataLoader(dataset, batch_size=8, num_workers=0)


def main() -> None:
    """Runs the playground in the foreground: prints usage instructions, then trains
    until `Trainer.fit` returns (Ctrl+C stops it gracefully via Lightning's own signal
    handling, since this runs on the main thread).
    """
    _init_single_process_distributed()

    tb_logger = TensorBoardLogger(save_dir=str(LOG_DIR), name="playground")
    print(_banner(Path(tb_logger.log_dir)))

    model = PlaygroundModel()
    trainer = pl.Trainer(
        max_epochs=-1,
        max_time="00:01:00:00",
        logger=tb_logger,
        enable_checkpointing=False,
        accelerator="cpu",
    )
    try:
        trainer.fit(model, train_dataloaders=_build_loader())
    finally:
        _teardown_single_process_distributed()
    print("Stopped.")


if __name__ == "__main__":
    main()
