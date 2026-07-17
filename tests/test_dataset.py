import json
from pathlib import Path

import torch
from PIL import Image

from src.dataset import get_transforms, load_class_names, save_class_names


def test_train_transform_output_shape():
    transform = get_transforms(train=True)
    dummy_image = Image.new("RGB", (256, 256))
    tensor = transform(dummy_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_eval_transform_output_shape():
    transform = get_transforms(train=False)
    dummy_image = Image.new("RGB", (300, 300))
    tensor = transform(dummy_image)
    assert tensor.shape == (3, 224, 224)


def test_save_and_load_class_names(tmp_path):
    path = tmp_path / "class_names.json"
    class_names = ["Tomato___healthy", "Tomato___Early_blight"]
    save_class_names(class_names, path=path)
    assert path.exists()
    loaded = load_class_names(path=path)
    assert loaded == class_names
    with open(path) as f:
        assert json.load(f) == class_names