# DeCLIP

A minimal implementation of "DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception" paper.

## What is DeCLIP?

CLIP struggles with dense prediction tasks (object detection, segmentation) due to the "proxy token" where attention focuses on random background patches instead of semantically related regions. DeCLIP fixes this by:

1. Decoupling the last attention layer into content (what) and context (where) features
2. Learning content features from CLIP itself via region-crop matching
3. Learning context features from Vision Foundation Models (DINO) that have better spatial understanding

## Requirements

```bash
pip install torch torchvision numpy pillow matplotlib datasets transformers timm tqdm
```

## Usage

```bash
python declip.py
```

This code will:
1. Load 1000 images from COCO validation set via HuggingFace
2. Show vanilla CLIP attention maps (notice scattered attention)
3. Train DeCLIP for 2 epochs (~10-15 minutes on GPU)
4. Show improved DeCLIP attention maps (focused on semantic regions)
5. Report region classification accuracy improvement

## Paper Reference

[DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](https://arxiv.org/pdf/2505.04410)