"""
Arsitektur model EfficientNetV2-S untuk klasifikasi biner single-view.

Backbone dimuat dari `timm` (pretrained ImageNet), classifier head diganti
dengan Dropout -> Linear(in_features, 2). Output 2 logit dipakai bersama
CrossEntropyLoss untuk kelas safe_driving dan phone_use. Probabilitas kelas
phone_use dihitung dengan softmax(logits)[:, 1] dan dipakai untuk
decision-level fusion (Persamaan 3.2-3.4 proposal).

Stage awal backbone dapat dibekukan sesuai konfigurasi eksperimen untuk
transfer learning yang lebih stabil.
"""

import logging

import timm
import torch.nn as nn

from src.config import MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def print_layer_status(model):
    """Cetak status pembekuan gradient per komponen model."""
    log.info("=" * 55)
    log.info("STATUS GRADIENT PARAMETER MODEL (FREEZE vs TRAINABLE)")
    log.info("=" * 55)

    stem_trainable = any(p.requires_grad for p in model.backbone.conv_stem.parameters())
    log.info("  backbone.conv_stem  : %s", "TRAINABLE" if stem_trainable else "FROZEN")

    bn1_trainable = any(p.requires_grad for p in model.backbone.bn1.parameters())
    log.info("  backbone.bn1        : %s", "TRAINABLE" if bn1_trainable else "FROZEN")

    for i, stage in enumerate(model.backbone.blocks):
        stage_trainable = any(p.requires_grad for p in stage.parameters())
        log.info(
            "  backbone.blocks[%d]  : %s (contains %d blocks)",
            i,
            "TRAINABLE" if stage_trainable else "FROZEN",
            len(stage),
        )

    head_trainable = any(p.requires_grad for p in model.backbone.conv_head.parameters())
    log.info("  backbone.conv_head  : %s", "TRAINABLE" if head_trainable else "FROZEN")

    bn2_trainable = any(p.requires_grad for p in model.backbone.bn2.parameters())
    log.info("  backbone.bn2        : %s", "TRAINABLE" if bn2_trainable else "FROZEN")

    classifier_trainable = any(p.requires_grad for p in model.classifier.parameters())
    log.info("  classifier          : %s", "TRAINABLE" if classifier_trainable else "FROZEN")
    log.info("=" * 55)


def build_model(pretrained: bool = True, num_stages_to_freeze: int = 5, dropout_rate: float = 0.5) -> nn.Module:
    """Bangun EfficientNetV2-S dengan classifier head 2 kelas.

    Parameters
    ----------
    pretrained : bool
        Jika True, muat bobot pretrained ImageNet. Set False saat load checkpoint.
    num_stages_to_freeze : int
        Jumlah stage block backbone yang dibekukan untuk transfer learning.
    dropout_rate : float
        Nilai dropout untuk classifier head.
    """
    backbone = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0)
    in_features = backbone.num_features

    classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 2),  # safe_driving, phone_use
    )

    model = _EfficientNetBinary(backbone, classifier)

    if num_stages_to_freeze > 0:
        log.info("Membekukan conv_stem, bn1, dan %d stage block pertama pada backbone...", num_stages_to_freeze)
        for p in backbone.conv_stem.parameters():
            p.requires_grad = False
        for p in backbone.bn1.parameters():
            p.requires_grad = False

        for i in range(min(num_stages_to_freeze, len(backbone.blocks))):
            for p in backbone.blocks[i].parameters():
                p.requires_grad = False

    print_layer_status(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s | Total params: %s | Trainable: %s", MODEL_NAME, f"{total_params:,}", f"{trainable_params:,}")
    return model


class _EfficientNetBinary(nn.Module):
    """Wrapper backbone feature extractor dan classifier biner 2-logit."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    import torch

    model = build_model(pretrained=True)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Output values: {out.tolist()}")
