"""Safe points at which TrainctlMixin may execute due commands or enter a hold."""

from enum import StrEnum


class SafePoint(StrEnum):
    """A Lightning lifecycle point legal for command execution and holds."""

    TRAIN_BATCH_END = "train_batch_end"
    TRAIN_EPOCH_END = "train_epoch_end"
    VALIDATION_EPOCH_END = "validation_epoch_end"
    BEFORE_OPTIMIZER_STEP = "before_optimizer_step"
    FIT_END = "fit_end"
