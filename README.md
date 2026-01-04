# MedKAN-MoE: Robust & Interpretable Medical Diagnostics 🏥🧠

![Robustness Graph](https://raw.githubusercontent.com/Saurabsingh778/MedKAN-MoE/main/results/robustness_curve.png)
*Figure: MedKAN-MoE outperforms traditional MLP baselines by a significant margin under high-noise conditions.*

## 🚀 Overview

**MedKAN-MoE** represents a paradigm shift in medical AI architectures. It combines the scalability of **Mixture of Experts (MoE)** with the mathematical transparency of **Kolmogorov-Arnold Networks (KANs)**.

Unlike traditional Multi-Layer Perceptrons (MLPs)—which act as opaque "black boxes" and rely on piecewise linear approximations (ReLU)—KANs utilize learnable B-spline activation functions on edges. This allows the network to learn the **exact mathematical structure** of the data.

Our results confirm that at equal parameter counts (~30k), **KANs are architecturally superior** for complex medical reasoning tasks.

---

## 🏆 Key Research Results

### 1. The "Iso-Parameter" Benchmark
We compared MedKAN-MoE against a standard MoE-MLP baseline where both models were restricted to **exactly ~30,000 parameters**. 

| Architecture | Parameters | Accuracy | Status |
| :--- | :--- | :--- | :--- |
| **MoE-KAN (Ours)** | **~30k** | **94.38%** | 🏆 **SOTA** |
| MoE-MLP (Baseline) | ~30k | 67.52% | |
| **Performance Gap** | | **+26.86%** | |

> **Key Finding:** KANs capture complex feature dependencies 26% better than MLPs given the same computational budget.

### 2. Robustness Stress Testing (Oracle Test)
Medical data is rarely clean. We subjected both models to Gaussian noise injection ($\sigma$). Even when the Router was given "perfect" information (Oracle Routing), the internal MLP experts performance degraded significantly compared to KAN experts.

| Noise Level ($\sigma$) | KAN Accuracy | MLP Accuracy | Advantage |
| :--- | :--- | :--- | :--- |
| **0.00 (Clean)** | **93.46%** | 68.00% | **+25.5%** |
| **0.01 (Low)** | **87.62%** | 60.30% | **+27.3%** |
| **0.03 (Med)** | **47.74%** | 31.72% | **+16.0%** |
| **0.05 (High)** | **21.20%** | 15.14% | **+6.1%** |

---

## 🧬 The Data: SNOWMED-ICD10
The model was trained on **SNOWMED-ICD10**, a custom-generated synthetic medical dataset rooted in **SNOMED-CT** and **ICD-10** ontologies. It tests three distinct levels of cognitive reasoning:

1. **Direct Mapping:** Clinical finding $\rightarrow$ ICD Code.
2. **Noisy Robustness:** Clinical notes with simulated typos and data entry errors.
3. **Needle-in-a-Haystack:** Diagnostic signals buried inside long, irrelevant administrative text.

**Preprocessing:** Input text was embedded via `all-MiniLM-L6-v2` and projected to a 20-dim PCA latent space.

---

## 📊 Visualizing the "Glass Box"

### 1. Learned B-Spline Functions
Instead of static weights, the KAN expert learns non-linear functions $\phi(x)$. Below is the learned shape for Feature 18, showing how the model reacts to input intensity.

![Learned Function](https://raw.githubusercontent.com/Saurabsingh778/MedKAN-MoE/main/results/spline_shape_expert_0_feat_18.png)

### 2. Native Feature Importance
The architecture provides intrinsic feature attribution scores without needing SHAP or LIME, ensuring clinician-level trust.

![Feature Importance](https://raw.githubusercontent.com/Saurabsingh778/MedKAN-MoE/main/results/feature_importance_expert_0.png)

---

## 🛠️ Architecture

* **Router:** A lightweight Gating Network that routes input samples to the most relevant Experts.
* **Experts:** `31` independent Kolmogorov-Arnold Networks (KANs), specialized by ICD chapters.
* **KAN Layers:** Replaces linear weights $W \cdot x$ with learnable 1D functions $\phi(x)$ parametrized as B-splines.

---

## 💻 Installation & Usage

### Setup
```bash
git clone [https://github.com/Saurabsingh778/MedKAN-MoE.git](https://github.com/Saurabsingh778/MedKAN-MoE.git)
cd MedKAN-MoE
pip install -r requirements.txt