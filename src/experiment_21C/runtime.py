from __future__ import annotations

import importlib

from src.experiment_21C.config_experiment_21C import FRAME_STRIDE_EXP21C


def apply_experiment_21C_runtime() -> None:
    """Patch loaded modules so Exp 21C consistently uses stride 20."""
    for module_name in ["src.config", "src.dataset", "src.fusion", "src.final_v1.dataset_final_v1"]:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        if hasattr(module, "FRAME_STRIDE"):
            setattr(module, "FRAME_STRIDE", FRAME_STRIDE_EXP21C)

