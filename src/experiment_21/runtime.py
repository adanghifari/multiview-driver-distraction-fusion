from __future__ import annotations

import importlib

from src.experiment_21.config_experiment_21 import FRAME_STRIDE_EXP21


def apply_experiment_21_runtime() -> None:
    """Patch loaded repo modules so Exp 21 consistently uses frame stride 20.

    The legacy training/evaluation modules read FRAME_STRIDE as an imported
    module constant. Patching both src.config and already-loaded consumers keeps
    the experiment reproducible without changing the global final_v1 default.
    """
    module_names = [
        "src.config",
        "src.dataset",
        "src.fusion",
        "src.final_v1.dataset_final_v1",
    ]
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        if hasattr(module, "FRAME_STRIDE"):
            setattr(module, "FRAME_STRIDE", FRAME_STRIDE_EXP21)

