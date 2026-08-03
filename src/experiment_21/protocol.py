def assert_checkpoint_frame_stride(checkpoint: dict, checkpoint_path: str, expected_stride: int, label: str) -> None:
    hyperparameters = checkpoint.get("hyperparameters") or {}
    actual_stride = hyperparameters.get("frame_stride")
    if actual_stride != expected_stride:
        raise ValueError(
            f"{label} checkpoint frame_stride mismatch for {checkpoint_path}: "
            f"expected {expected_stride}, got {actual_stride}. "
            "Retrain this experiment before evaluation so the protocol stays apple-to-apple."
        )
