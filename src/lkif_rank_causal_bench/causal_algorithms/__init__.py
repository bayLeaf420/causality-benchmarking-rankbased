"""Causal algorithms behind one interface, built by name from their YAML specs."""

from typing import Any

from lkif_rank_causal_bench.causal_algorithms.base import CausalAlgorithm, Estimate
from lkif_rank_causal_bench.causal_algorithms.ccc import CCC
from lkif_rank_causal_bench.causal_algorithms.ccm import CCM
from lkif_rank_causal_bench.causal_algorithms.granger import GrangerCausality
from lkif_rank_causal_bench.causal_algorithms.pcmci import PCMCI
from lkif_rank_causal_bench.causal_algorithms.transfer_entropy import KraskovTE

REGISTRY: dict[str, type[CausalAlgorithm]] = {cls.name: cls for cls in (GrangerCausality, KraskovTE, CCC, PCMCI, CCM)}


def build_algorithm(name: str, params: dict[str, Any]) -> CausalAlgorithm:
    """Instantiate a registered algorithm with its hyperparameters."""
    return REGISTRY[name](**params)


__all__ = [
    "CCC",
    "CCM",
    "PCMCI",
    "REGISTRY",
    "CausalAlgorithm",
    "Estimate",
    "GrangerCausality",
    "KraskovTE",
    "build_algorithm",
]
