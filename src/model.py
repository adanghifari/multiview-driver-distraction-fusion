"""
Arsitektur model EfficientNetV2-S untuk klasifikasi biner single-view.

Backbone dimuat dari `timm` (pretrained ImageNet), classifier head diganti
dengan Dropout → Linear(in_features, 1). Output 1 neuron (logit) karena
menggunakan BCEWithLogitsLoss — skor sigmoid-nya nanti langsung bisa dipakai
sebagai confidence score untuk decision-level fusion (Persamaan 3.2–3.4
proposal).

Semua parameter backbone di-unfreeze → full fine-tuning sesuai Subbab 3.3
proposal.
"""

import logging

import timm
import torch.nn as nn

from src.config import MODEL_NAME, DROPOUT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def build_model(pretrained: bool = True) -> nn.Module:
    """Bangun EfficientNetV2-S dengan classifier head untuk klasifikasi biner.

    Parameters
    ----------
    pretrained : bool
        Jika True, muat bobot pretrained ImageNet. Set False jika ingin
        load dari checkpoint sendiri.

    Returns
    -------
    nn.Module
        Model siap training / inference.
    """
    backbone = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0)
    in_features = backbone.num_features

    classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 1),
    )

    model = _EfficientNetBinary(backbone, classifier)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s | Total params: %s | Trainable: %s",
             MODEL_NAME, f"{total_params:,}", f"{trainable_params:,}")
    return model


class _EfficientNetBinary(nn.Module):
    """Wrapper: backbone (feature extractor) + custom binary classifier head."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)     # (B, in_features)
        logits = self.classifier(features)  # (B, 1)
        return logits.squeeze(1)        # (B,) — agar cocok dengan BCEWithLogitsLoss


if __name__ == "__main__":
    import torch

    model = build_model(pretrained=True)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")     # Harusnya torch.Size([2])
    print(f"Output values: {out.tolist()}")
