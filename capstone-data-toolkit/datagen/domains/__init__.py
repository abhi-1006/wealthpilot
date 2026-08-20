"""Domain generators, one per capstone problem statement."""

from .base import DocSpec, DomainSpec, EvalCase
from .careflow import CareFlow
from .lexops import LexOps
from .plantguard import PlantGuard
from .shopsense import ShopSense
from .wealthpilot import WealthPilot

REGISTRY: dict[str, type[DomainSpec]] = {
    CareFlow.key: CareFlow,
    LexOps.key: LexOps,
    WealthPilot.key: WealthPilot,
    ShopSense.key: ShopSense,
    PlantGuard.key: PlantGuard,
}

__all__ = [
    "DocSpec",
    "DomainSpec",
    "EvalCase",
    "REGISTRY",
    "CareFlow",
    "LexOps",
    "WealthPilot",
    "ShopSense",
    "PlantGuard",
]
