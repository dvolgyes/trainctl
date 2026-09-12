"""Lightning namespace detection and lazy backend loading for TrainctlMixin."""

import importlib
from dataclasses import dataclass
from types import ModuleType

LIGHTNING_PYTORCH = "lightning.pytorch"
PYTORCH_LIGHTNING = "pytorch_lightning"
_KNOWN_FAMILIES = (LIGHTNING_PYTORCH, PYTORCH_LIGHTNING)


@dataclass(frozen=True)
class LightningBackend:
    """The Lightning namespace a TrainctlMixin subclass was composed with.

    Attributes:
        name: Backend family identifier (`lightning.pytorch` or `pytorch_lightning`).
        PL: The imported Lightning package module.
        LightningModule: The backend's `LightningModule` base class.
        Trainer: The backend's `Trainer` class.
        version: The backend package's reported version string.
    """

    name: str
    PL: ModuleType
    LightningModule: type
    Trainer: type
    version: str


def _family_of(module_name: str) -> str | None:
    for family in _KNOWN_FAMILIES:
        if module_name == family or module_name.startswith(family + "."):
            return family
    return None


def detect_lightning_backend_from_mro(mro: tuple[type, ...]) -> LightningBackend:
    """Determines the Lightning family represented anywhere in a class MRO.

    Args:
        mro: The `__mro__` of the class being defined.

    Returns:
        The loaded backend for the single Lightning family found in the MRO.

    Raises:
        TypeError: No known Lightning family is present, or more than one is.
    """
    families: set[str] = set()
    for base in mro:
        family = _family_of(base.__module__)
        if family is not None:
            families.add(family)

    if not families:
        raise TypeError(
            "TrainctlMixin must be combined with either lightning.pytorch.LightningModule "
            "or pytorch_lightning.LightningModule"
        )
    if len(families) != 1:
        raise TypeError(
            "A Trainctl model cannot mix lightning.pytorch and pytorch_lightning classes "
            f"in the same inheritance tree (found: {sorted(families)})"
        )
    return load_backend(families.pop())


def load_backend(name: str) -> LightningBackend:
    """Lazily imports and describes one Lightning backend family.

    Args:
        name: `lightning.pytorch` or `pytorch_lightning`.

    Returns:
        The loaded backend descriptor.

    Raises:
        ValueError: `name` is not a known backend family.
    """
    if name not in _KNOWN_FAMILIES:
        raise ValueError(f"Unknown Lightning backend {name!r}")
    pl_module = importlib.import_module(name)
    return LightningBackend(
        name=name,
        PL=pl_module,
        LightningModule=pl_module.LightningModule,
        Trainer=pl_module.Trainer,
        version=getattr(pl_module, "__version__", "unknown"),
    )
