from src.model import build_model


def build_final_model(pretrained: bool = True, num_stages_to_freeze: int = 5, dropout_rate: float = 0.5):
    """Build the final_v1 EfficientNetV2-S model using the repo's stable implementation."""
    return build_model(
        pretrained=pretrained,
        num_stages_to_freeze=num_stages_to_freeze,
        dropout_rate=dropout_rate,
    )
