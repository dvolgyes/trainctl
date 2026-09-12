"""Shared availability discovery for the optional-package inspectors.

One place computing `{name: {"available": bool}}`, called from both `GET /inspectors`
and the FUSE `/model/inspectors.json` projection so availability logic is never
duplicated between the two surfaces.
"""

from trainctl.runtime import graph_inspectors, torchinfo_inspector, torchlens_inspector


def availability_snapshot() -> dict[str, dict[str, bool]]:
    """Returns each optional-package inspector's current availability."""
    return {
        "torchinfo": {"available": torchinfo_inspector.available()},
        "torchview": {"available": graph_inspectors.torchview_available()},
        "torchviz": {"available": graph_inspectors.torchviz_available()},
        "torchlens": {"available": torchlens_inspector.available()},
    }
