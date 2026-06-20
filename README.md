# FruArIC
### Frugal Architecture for Identifying Container Codes in Port Automation

Official implementation of the paper:

> **FruArIC: Frugal Architecture for Identifying Container Codes in Port Automation**

---

# Architecture

![Architecture](assets/architecture.png)

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

### Compared with YOLOv12 + SRN

- 98.6% fewer parameters
- 72.5% fewer MACs
- 77.4% lower latency
- Only 1.9 percentage-point accuracy difference

---

# Dataset

| Split | Images |
|---------|---------:|
| Train | 15,128 |
| Validation | 1,892 |
| Test | 1,891 |
| Independent Evaluation | 1,000 |

---

# Installation

```bash
conda create -n fruaric python=3.10 -y
conda activate fruaric

pip install -r requirements.txt
```

---

# Training

## TailoredDet

```bash
python train_detector.py
```

## PureConvSTR

```bash
python train_recognizer.py
```

---

# Inference

```bash
python inference.py --image sample.jpg --weights weights/fruaric.pt
```

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

![Results](assets/results_table.png)

---

# Ablation Studies

### TailoredDet

- P3-only detection branch
- P5 backbone removal
- Cross-YOLO validation

### PureConvSTR

- Hierarchical knowledge distillation
- Distillation vs Pruning vs QAT

---

# Citation

```bibtex
@article{kang2026fruaric,
  title={FruArIC: Frugal Architecture for Identifying Container Codes in Port Automation}
}
```

---

# Acknowledgement

The recognition module is adapted from the Deep Text Recognition Benchmark developed by Clova AI and modified for container code recognition.
