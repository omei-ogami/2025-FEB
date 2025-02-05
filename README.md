# Satellite Images for Disaster Recognition

This is the side project for 2025 Febuary. This project aims to assess building damage from satellite imagery using deep learning. It is based on the xView2 dataset, where the model predicts damage levels by comparing pre-disaster and post-disaster images.

## Folder Organization
```bash
/2025-FRB
│── /data               # Ignored by .gitignore
│   │── /train
│   │   │── /images
│   │   │   │── /pre
│   │   │   │   │── disaster_0001_pre.png
│   │   │   │   │── disaster_0002_pre.png
│   │   │   │── /post
│   │   │   │   │── disaster_0001_post.png
│   │   │   │   │── disaster_0002_post.png
│   │   │── /targets
│   │   │   │── disaster_0001_target.png
│   │   │   │── disaster_0002_target.png
│   │── /val
│   │   │── /images
│   │   │   │── pre
│   │   │   │── post
│   │   │── /targets
│   │── /test
│       │── /images
│       │   │── pre
│       │   │── post
│       │── /targets
│── /src                # Codes (data loader, model, training script, etc.)
│── /models             # Trained models
│── .gitignore          # Prevents large files from being pushed
│── README.md           # Documentation
```
