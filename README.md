# Physics-Informed Fourier Neural Operators for Tokamak MHD Tearing Mode Forecasting

## Overview
This repository contains a high-performance computational framework for predicting Magnetohydrodynamic (MHD) instabilities—specifically tearing modes and magnetic island formation—in tokamak fusion plasmas. 

Standard deep learning approaches (such as CNNs or black-box PINNs) suffer from mesh dependence and high computational latency. This project implements a **Fourier Neural Operator (FNO)** regularized by **Reduced Magnetohydrodynamic (RMHD) conservation laws** to achieve resolution-independent, sub-millisecond forecasting of plasma disruption precursors.

---

## Repository Structure
```text
Tokamak-MHD-FNO/
│
├── data/
│   └── data_generator.py     # Synthetic RMHD telemetry & magnetic flux generator
├── models/
│   └── fno_model.py          # 2D Fourier Neural Operator (SpectralConv2d & TokamakFNO2d)
├── utils/
│   └── physics_loss.py       # Physics-informed RMHD energy & flux residual loss
├── main.py                   # Unified training loop, evaluation, and latency benchmark
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation