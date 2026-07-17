"""Usage: python -m src.predict --image path/to/leaf.jpg"""

import argparse

import torch
from PIL import Image

from src import config
from src.dataset import get_transforms, load_class_names
from src.model import load_model


def predict_image(image_path: str, device: str = "cpu") -> list:
    class_names = load_class_names()
    model = load_model(len(class_names), config.MODEL_PATH, device=device)

    transform = get_transforms(train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=min(3, len(class_names)))
    return [(class_names[i], float(p)) for p, i in zip(top_probs, top_idxs)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for label, prob in predict_image(args.image, device=device):
        print(f"  {label}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()