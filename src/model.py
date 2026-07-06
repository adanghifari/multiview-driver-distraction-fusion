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

from src.config import MODEL_NAME, DROPOUT, NUM_STAGES_TO_FREEZE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def print_layer_status(model):
    """Cetak status pembekuan (gradient requirements) per komponen model secara terstruktur."""
    log.info("=" * 55)
    log.info("STATUS GRADIENT PARAMETER MODEL (FREEZE vs TRAINABLE)")
    log.info("=" * 55)
    
    stem_trainable = any(p.requires_grad for p in model.backbone.conv_stem.parameters())
    log.info("  backbone.conv_stem  : %s", "TRAINABLE" if stem_trainable else "FROZEN")
    
    bn1_trainable = any(p.requires_grad for p in model.backbone.bn1.parameters())
    log.info("  backbone.bn1        : %s", "TRAINABLE" if bn1_trainable else "FROZEN")
    
    for i, stage in enumerate(model.backbone.blocks):
        stage_trainable = any(p.requires_grad for p in stage.parameters())
        log.info("  backbone.blocks[%d]  : %s (contains %d blocks)", 
                 i, "TRAINABLE" if stage_trainable else "FROZEN", len(stage))
                 
    head_trainable = any(p.requires_grad for p in model.backbone.conv_head.parameters())
    log.info("  backbone.conv_head  : %s", "TRAINABLE" if head_trainable else "FROZEN")
    
    bn2_trainable = any(p.requires_grad for p in model.backbone.bn2.parameters())
    log.info("  backbone.bn2        : %s", "TRAINABLE" if bn2_trainable else "FROZEN")
    
    classifier_trainable = any(p.requires_grad for p in model.classifier.parameters())
    log.info("  classifier          : %s", "TRAINABLE" if classifier_trainable else "FROZEN")
    log.info("=" * 55)


def build_model(pretrained: bool = True, num_stages_to_freeze: int = NUM_STAGES_TO_FREEZE) -> nn.Module:
    """Bangun EfficientNetV2-S dengan classifier head untuk klasifikasi biner (2-unit output).

    Parameters
    ----------
    pretrained : bool
        Jika True, muat bobot pretrained ImageNet. Set False jika ingin
        load dari checkpoint sendiri.
    num_stages_to_freeze : int
        Jumlah stage block backbone yang dibekukan (frozen) untuk transfer learning.

    Returns
    -------
    nn.Module
        Model siap training / inference.
    """
    backbone = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0)
    in_features = backbone.num_features

    classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 2),  # 2 output units (safe_driving, phone_use)
    )

    model = _EfficientNetBinary(backbone, classifier)

    # Bekukan stage awal backbone jika dikonfigurasi
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
        logits = self.classifier(features)  # (B, 2)
        return logits                   # (B, 2) — cocok untuk CrossEntropyLoss


if __name__ == "__main__":
    import torch

    model = build_model(pretrained=True)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")     # Harusnya torch.Size([2, 2])
    print(f"Output values: {out.tolist()}")

