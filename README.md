# FruArIC
### Frugal Architecture for Identifying Container Codes in Port Automation

Official implementation of the paper:

> **FruArIC: Frugal Architecture for Identifying Container Codes in Port Automation**

---

# Architecture

<img width="1412" height="1566" alt="Image" src="https://github.com/user-attachments/assets/7c61718b-2850-48a9-89f2-143acfe99298" />

FruArIC is a lightweight two-stage framework designed for container code recognition in resource-constrained edge environments.

- **TailoredDet**: P3-centric lightweight detector
- **PureConvSTR**: Convolution-only recognizer trained through hierarchical knowledge distillation
- **Format-Constrained Post-processing**

---

# Highlights

| Metric | FruArIC |
|----------|----------|
| Full-string Accuracy | **90.233%** |
| Parameters | **0.816 M** |
| MACs | **2.118 G** |
| CPU Latency | **13.955 ms** |

---

# Dataset

| Split | Images |
|---------|---------:|
| Train | 15,128 |
| Validation | 1,892 |
| Test | 1,891 |
| Independent Evaluation | 1,000 |

---


# Implementation Details

### TailoredDet

- Input size: 640×640 RGB
- Batch size: 16
- Learning rate: 0.01
- Optimizer: MuSGD
- Losses: CIoU, DFL, BCE

### PureConvSTR

- Input size: 150×45 Grayscale
- Batch size: 192
- Learning rate: 1.0
- Optimizer: Adadelta
- Recognition loss: CTC
- Distillation loss: KL Divergence

### Hardware

- Training: NVIDIA GeForce RTX 5080
- Inference: Intel Ultra 7 265K CPU

---

# Experimental Results

<img width="692" height="449" alt="Image" src="https://github.com/user-attachments/assets/70fb6961-e38e-4006-9865-650469060d4a" />

---


# Usage

Run `main.py` with an input image and the pretrained model weights.
```bash
python main.py \
  --image_path ./sample.jpg \
  --detector_path ./weights/tailoreddet.pt \
  --recognizer_path ./weights/pureconvstr.pth \
  --FeatureExtraction CNN_s \
  --input_channel 1 \
  --output_channel 512 \
  --padding 4
```


---
