"""Standalone, fast Lightning training run launched as a subprocess by
`tests/integration/parallel_runs_test.py` to exercise Trainctl's REST and
torch_debug port-collision avoidance, and MASTER_PORT isolation for
`torch.distributed`, across two simultaneous runs.

Not a pytest test module -- deliberately does not match pytest's collection
glob (`test_*.py` / `*_test.py`). Run directly:

    TRAINCTL_TEST_MASTER_PORT=... TRAINCTL_TEST_LOG_DIR=... \\
        TRAINCTL_TEST_STATUS_FILE=... python _parallel_training_fixture.py

Trains with `max_epochs=-1` on a tiny synthetic dataset and relies entirely on
an external `POST /control/stop` (or the `max_time` safety net below) to end
the run -- so a caller has a real, indefinitely-running instance to poll.

Environment variables:
    TRAINCTL_TEST_MASTER_PORT: Required. Distinct `torch.distributed`
        rendezvous port for this instance -- trainctl has no collision
        avoidance for this port, so each parallel instance must supply its own.
    TRAINCTL_TEST_LOG_DIR: Required. This instance's artifact directory.
    TRAINCTL_TEST_STATUS_FILE: Required. Path this instance writes its
        resolved run_id/rest_port/torch_debug_url to, once its services start.
    TRAINCTL_TEST_REST_PORT: Optional, default 8090 (trainctl's own default)
        -- deliberately left shared across instances, so the test exercises
        trainctl's preflight REST port search under real contention.
    TRAINCTL_TEST_TORCH_DEBUG_PORT: Optional, default 25999, same rationale.
"""

import json
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import v2 as T

from trainctl import TrainctlMixin


class _FastModel(TrainctlMixin, pl.LightningModule):
    """Tiny classifier used only to exercise Trainctl's service startup.

    Attributes:
        net: Small linear classifier over flattened 8x8 RGB inputs.
    """

    def __init__(
        self,
        rest_port: int,
        torch_debug_port: int,
        artifact_dir: Path,
        status_file: Path,
    ) -> None:
        super().__init__(
            rest_port=rest_port,
            torch_debug_port=torch_debug_port,
            fuse_enabled=False,
            hooks_enabled=False,
            artifact_dir=artifact_dir,
            inspection_strict=True,
        )
        self._status_file = status_file
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(3 * 8 * 8, 8), nn.ReLU(), nn.Linear(8, 4)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.net(images)
        return logits

    def training_step(  # pylint: disable=arguments-differ
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        images, labels = batch
        return nn.functional.cross_entropy(self(images), labels)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def setup(self, stage: str) -> None:
        super().setup(stage)
        snapshot = self._trainctl.state.read()
        if snapshot.rest_port is not None:
            self._status_file.write_text(
                json.dumps(
                    {
                        "run_id": snapshot.run_id,
                        "rest_port": snapshot.rest_port,
                        "torch_debug_url": snapshot.torch_debug_url,
                        "artifacts_path": snapshot.artifacts_path,
                    }
                )
            )


def _init_distributed(master_port: str) -> None:
    """Starts a real, single-rank `torch.distributed` process group on `master_port`."""
    import torch.distributed as dist

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _build_loader() -> DataLoader:
    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    dataset = FakeData(size=8, image_size=(3, 8, 8), num_classes=4, transform=transform)
    return DataLoader(dataset, batch_size=4, num_workers=0)


def main() -> None:
    import torch.distributed as dist

    master_port = os.environ["TRAINCTL_TEST_MASTER_PORT"]
    log_dir = Path(os.environ["TRAINCTL_TEST_LOG_DIR"])
    status_file = Path(os.environ["TRAINCTL_TEST_STATUS_FILE"])
    rest_port = int(os.environ.get("TRAINCTL_TEST_REST_PORT", "8090"))
    torch_debug_port = int(os.environ.get("TRAINCTL_TEST_TORCH_DEBUG_PORT", "25999"))

    _init_distributed(master_port)
    try:
        model = _FastModel(rest_port, torch_debug_port, log_dir, status_file)
        trainer = pl.Trainer(
            max_epochs=-1,
            max_time="00:00:01:00",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
        )
        trainer.fit(model, train_dataloaders=_build_loader())
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
