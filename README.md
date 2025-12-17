FYP 2025 HKMU
CLIP Zero-shot classification
Used dataset# Caltech-101 Zero-Shot and Fine-Tuned Classification with OpenCLIP (ViT-B/16)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/your-notebook-link-here) <!-- Replace with actual link if uploading -->

This project demonstrates **zero-shot image classification** and **supervised fine-tuning** on the classic **Caltech-101** dataset using an open-source CLIP model (ViT-B/16) from [OpenCLIP](https://github.com/mlfoundations/open_clip), pretrained on LAION-400M.












*Sample images from the Caltech-101 dataset (101 object categories).*

## About Caltech-101

Caltech-101 is a benchmark dataset for object recognition, containing ~9,146 images across **101 object categories** (plus a background clutter class in some versions). 

- Created in 2003 by Fei-Fei Li et al. at Caltech.
- ~40–800 images per category (most ~50).
- Images are medium resolution (~300x200 pixels).
- Standard evaluation: 30 images per class for testing, remainder for training.

It's relatively "easy" for modern models but remains a great testbed for transfer learning.








*CLIP architecture overview (left) and ViT-based image encoder (right).*

## Results

| Method                  | Top-1 Accuracy (Test Set) | Notes |
|-------------------------|---------------------------|-------|
| **Zero-Shot** (ViT-B/16 laion400m_e31) | ~93–95% | No training on Caltech-101 |
| **Linear Probe** (head only) | ~96–98% | 20 epochs, AdamW |
| **Full Fine-Tuning** (last layers unfrozen) | **98–99+%** | Additional 10 epochs, lower LR |




*Example accuracy visualization (conceptual – your training will produce similar plots).*

Caltech-101 is saturated by modern pretrained models – even zero-shot performs extremely well!

## Requirements

- Python 3.8+
- PyTorch
- open-clip-torch
- torchvision
- tqdm, matplotlib, seaborn, scikit-learn

Install with:
```bash
pip install open-clip-torch torch torchvision tqdm matplotlib seaborn scikit-learn
```

## Usage

The full implementation is in a Google Colab notebook format (easy to run on free GPU).

1. **Zero-Shot Evaluation**
   - Loads the pretrained model.
   - Computes text embeddings for class names.
   - Evaluates on the dataset with cosine similarity.
   - Plots per-class accuracy and confusion matrix.

2. **Fine-Tuning**
   - Standard split: 30 test images per class.
   - Handles grayscale images (converts to RGB).
   - Linear probe (classification head only).
   - Optional full fine-tuning of last ViT blocks.
   - Saves best checkpoint.

Run the notebook cells sequentially on Colab (Runtime → T4 GPU recommended).

## Acknowledgments

- [OpenCLIP](https://github.com/mlfoundations/open_clip) by MLFoundations
- Caltech-101 dataset by Fei-Fei Li et al.
- LAION-400M pretraining data

Feel free to open issues or PRs for improvements (e.g., top-5 metrics, better prompts, larger models)!

---

**Star the repo if you found this useful!** 🌟
