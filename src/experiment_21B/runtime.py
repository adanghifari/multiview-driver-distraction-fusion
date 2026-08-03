from __future__ import annotations

import importlib

from src.experiment_21B.config_experiment_21B import FRAME_STRIDE_EXP21B


def apply_experiment_21B_runtime() -> None:
    """Patch loaded modules so Exp 21B consistently uses stride 20."""
    for module_name in ["src.config", "src.dataset", "src.fusion", "src.final_v1.dataset_final_v1"]:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        if hasattr(module, "FRAME_STRIDE"):
            setattr(module, "FRAME_STRIDE", FRAME_STRIDE_EXP21B)

