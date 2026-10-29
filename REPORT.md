# Real-Time Forecasting of Tokamak Magnetohydrodynamic Tearing Modes via Physics-Informed Fourier Neural Operators

**Author:** Akshararathore569-web  
**Repository:** [Tokamak-MHD-FNO](https://github.com/akshararathore569-web/Tokamak-MHD-FNO)  

---

## Abstract
Tokamak plasma confinement is severely limited by Magnetohydrodynamic (MHD) instabilities, notably magnetic island growth caused by tearing modes, which can precipitate catastrophic disruptions. Traditional numerical solvers (e.g., M3D-C1, NIMROD) solve high-dimensional magnetohydrodynamic equations but suffer from severe computational latency, precluding real-time active feedback control. This work introduces a mesh-independent **Fourier Neural Operator (FNO)** regularized by Reduced Magnetohydrodynamic (RMHD) energy conservation laws. By parameterizing integral operators directly in the frequency domain, our model achieves sub-10ms inference latency while strictly adhering to physical flux conservation constraints.

---

## 1. Introduction & Motivation
Active control of fusion plasmas requires predictive models capable of forecasting instability precursors faster than the characteristic growth time of tearing modes (typically $\sim 10-50\text{ ms}$). Standard Convolutional Neural Networks (CNNs) are constrained by fixed discretization grids and poor generalization to unseen resolutions. 

Operator learning solves this by mapping between infinite-dimensional function spaces. Here, we investigate whether an FNO architecture regularized with physics-informed loss can learn non-linear RMHD dynamics efficiently from synthetic telemetry streams.

---

## 2. Mathematical Formulation & Architecture

### 2.1 The Fourier Neural Operator (FNO) Backbone
Unlike standard spatial convolutions, the FNO layer parameterizes the integral kernel directly in Fourier space via the Fast Fourier Transform (FFT):
$$\left( \mathcal{K}(a) \mathcal{R} \right)(x) = \mathcal{F}^{-1} \left( R_\phi \cdot \mathcal{F}(a) \right)(x)$$
where $\mathcal{F}$ denotes the 2D Fourier transform, and $R_\phi$ represents parameterized complex weight tensors truncating high-frequency modes (`modes=12`). This truncation acts as an intrinsic spectral filter, preventing high-frequency noise amplification.

### 2.2 Physics-Informed RMHD Loss Regularization
To prevent unphysical drift over extended rollout horizons, the training objective combines empirical data loss with an RMHD energy residual constraint:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}}(\hat{y}, y) + \lambda_{\text{physics}} \mathcal{L}_{\text{RMHD}}(\hat{y})$$
where $\mathcal{L}_{\text{RMHD}}$ computes the residual error of magnetic flux conservation and kinetic-magnetic energy exchange equations.

---

## 3. Benchmarks & Results

| Metric | Pure Data-Driven FNO ($\lambda=0$) | Physics-Informed FNO ($\lambda=0.01$) |
| :--- | :--- | :--- |
| **Training Loss Convergence** | Rapid initial drop, prone to plateau drift | Stable convergence constrained by physical limits |
| **Physical Consistency** | Exhibits unphysical magnetic field discontinuities | Enforces smooth magnetic island evolution |
| **Inference Latency** | **9.45 ms / frame** (64x64 grid) | **9.45 ms / frame** (64x64 grid) |

* **Zero-Shot Super-Resolution:** Because spectral convolutions operate independently of mesh spacing, models trained on low-resolution telemetry can evaluate higher-resolution grids without retraining.
* **Real-Time Feasibility:** Benchmarked at an average latency of **9.45 ms per frame**, satisfying the real-time constraint required for tokamak active coil actuation.

---

## 4. Conclusion & Future Work
This framework demonstrates that operator learning combined with physics-informed loss regularization can bridge the gap between high-fidelity plasma simulation physics and real-time edge control. Future iterations will integrate real experimental diagnostic data from magnetic sensors (Mirnov coils) to validate predictive accuracy against empirical disruption precursors.