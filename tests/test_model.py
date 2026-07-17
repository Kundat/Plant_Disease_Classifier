import torch
from src.model import build_model


def test_build_model_output_shape():
    model = build_model(5, pretrained=False, freeze_backbone=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 5)


def test_build_model_freeze_backbone():
    model = build_model(3, pretrained=False, freeze_backbone=True)
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc")]
    head_params = [p for name, p in model.named_parameters() if name.startswith("fc")]
    assert all(not p.requires_grad for p in backbone_params)
    assert all(p.requires_grad for p in head_params)