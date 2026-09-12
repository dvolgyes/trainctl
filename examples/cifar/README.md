# CIFAR-10 integration harness

A small, real training workload for exercising trainctl's LightningModule-compatible base class against an actual
Lightning `Trainer`, callbacks, and TensorBoard logging — not a benchmark.

## Trainctl integration

`CifarClassifier` composes `TrainctlMixin` before `lightning.pytorch.LightningModule`:

```python
import lightning.pytorch as pl
from trainctl import TrainctlMixin


class CifarClassifier(TrainctlMixin, pl.LightningModule): ...
```

This starts a read-only inspection filesystem (mfusepy, default under `<TensorBoard log_dir>/procfs`, alongside this
run's checkpoints and TensorBoard logs) and a REST control API (default `http://127.0.0.1:8090`) around the run, with no
other changes to `data.py` or `train.py`. See `trainctl/mixin.py` for the full set of `trainctl_*` constructor options
(ports, mountpoint, enabling/disabling each service); the mounted filesystem's own `/README.md` documents its tree, and
`trainctl/rest/app.py` documents the REST routes.

While a run is active:

```bash
cat examples/cifar/lightning_logs/cifar/version_0/procfs/state/status
curl http://127.0.0.1:8090/run
curl -X POST http://127.0.0.1:8090/holds -d '{"when":{"safe_point":"train_epoch_end","occurrence":"next"}}'
```

## Running

```bash
uv run python -m examples.cifar.train
```

Defaults to a 1-hour wall-clock run (`--max-time 00:01:00:00`) with `--max-epochs -1`, so duration is governed by
`--max-time` rather than a fixed epoch count. See `uv run python -m examples.cifar.train --help` for all options (batch
size, learning rate, data/log directories, accelerator, etc.).

The first run downloads CIFAR-10 (~170 MB) into `examples/cifar/data/`; subsequent runs reuse the cached copy.
TensorBoard logs and checkpoints are written to `examples/cifar/lightning_logs/`. View them with:

```bash
uv run tensorboard --logdir examples/cifar/lightning_logs
```

Logged per run: `train/loss` and `val/loss` (loss), `train/acc` and `val/acc` (accuracy metric), and the optimizer
hyperparameters (`learning_rate`, `weight_decay`, `lr_step_size`, `lr_gamma`) plus a live learning-rate scalar via
`LearningRateMonitor`.

## Caveats

- The torchvision CIFAR-10 *test* split is used as the validation split. This is an integration/soak-test harness for
  the training loop, not a rigorous benchmark protocol.
- `examples/cifar/data/` and `examples/cifar/lightning_logs/` are gitignored.
