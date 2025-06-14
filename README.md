# OCR-with-multi-input-model-Development


This project is a multi-modal text image classifier that demonstrates a POC multi-modal classification approach to train an Optical Character Recognition (OCR) model. To improve the classification, the model will use images of the scanned documents as input and their insurance type (home, life, auto, health, or other). Integrating different data modalities (such as image and text) enables the model to perform better in complex scenarios, helping to capture more nuanced information. The labels that the model will be trained to identify are of two types: a primary and a secondary ID, for each image-insurance type pair. It combines image representations of text with one-hot encoded categorical vectors to predict associated labels.

## 📂 Project Structure

- `notebook.ipynb`: Main Jupyter notebook for data exploration, training, and evaluation.
- `project_utils.py`: Utility module that defines the `ProjectDataset` class, which generates synthetic data samples for training/testing.

## 📊 Problem Description

Each sample contains:
- A **64x64 grayscale image** of a random string (alphanumeric).
- A **type vector** indicating one of five categories: `home`, `life`, `auto`, `health`, `other`.

The task is to classify whether each sample is a `primary_id` or `secondary_id` based on the combined features.

## 🧱 Dataset

The dataset is fully synthetic and generated on-the-fly using the `ProjectDataset` class:
- **Images** are created from random 5-character strings.
- **Labels** (`primary_id` = 0, `secondary_id` = 1) are randomly assigned.
- **Type vectors** are one-hot encodings of randomly selected types.

## ⚙️ Dependencies

Make sure the following Python packages are installed:

```bash
torch
torchvision
Pillow
```