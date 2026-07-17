# 🌿 Plant Disease Classifier

A deep learning image classifier that identifies plant diseases from leaf photos, built with PyTorch using transfer learning on ResNet18. Includes a neon dark-themed Streamlit demo app for interactive predictions.

## Demo

Upload a leaf photo → get an instant diagnosis with confidence scores.
streamlit run app/streamlit_app.py
## Features

- Transfer learning on a pretrained ResNet18 backbone
- Data augmentation pipeline (random crop, flip, rotation, color jitter)
- Training script with CLI configuration and checkpointing on best validation accuracy
- Interactive Streamlit app with a custom neon dark theme
- Unit tests covering model architecture and data pipeline
- Trained on the PlantVillage dataset (15 classes, ~19,000 images)

## Results

| Metric | Value |
|---|---|
| Validation accuracy | 80.7% |
| Classes | 15 |
| Training epochs | 5 |
| Architecture | ResNet18 (transfer learning) |

## Project Structure

plant-disease-classifier/
├── src/
│   ├── config.py       # Paths, hyperparameters
│   ├── dataset.py      # Dataloaders, transforms, class name persistence
│   ├── model.py         # ResNet18-based model definition
│   ├── train.py         # Training loop with CLI args
│   └── predict.py       # Single-image inference
├── app/
│   └── streamlit_app.py # Interactive web demo
├── tests/
│   ├── test_model.py
│   └── test_dataset.py
├── split_data.py         # One-time script to split raw dataset into train/val
└── .streamlit/
└── config.toml       # Custom neon dark theme
## Setup

1. Clone the repo and install dependencies:

   ```bash
   git clone https://github.com/<Kundat>/plant-disease-classifier.git
   cd plant-disease-classifier
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
2. Download the PlantVillage dataset from Kaggle.
3. Split it into data/train/ and data/val/ folders (one subfolder per class) using split_data.py.
## Future Improvements

-Support more plant species and diseases
-Improve preddiction accuracy with additional training data
-Deploy the application online for public use
-Add treatment recommendaions foe detected diseases
## Author
** KUNDA TEMBO **
Computer Science Student | Machine learning & Software Development Enthusiast
