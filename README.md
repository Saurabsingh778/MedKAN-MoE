# MedKAN-MoE: Robust & Interpretable Medical Diagnostics 🏥🧠

![Robustness Graph](https://raw.githubusercontent.com/Saurabsingh778/MedKAN-MoE/main/results/robustness_curve.png)
*Figure: MedKAN-MoE outperforms traditional MLP baselines by a 27.8% accuracy margin under high-noise conditions.*

## 🚀 Overview

**MedKAN-MoE** combines the architectural efficiency of **Mixture of Experts (MoE)** with the mathematical interpretability of **Kolmogorov-Arnold Networks (KANs)**. 

While traditional Multi-Layer Perceptrons (MLPs) act as "black boxes" and degrade rapidly under data noise, KANs learn readable B-spline activation functions. This project demonstrates that KAN-based Experts are not only **more accurate** but significantly **more robust** to noise and distributional shifts than their MLP counterparts.

## ✨ Key Achievements

1.  **Unmatched Robustness:** 
    - Achieved a **+27.8% Accuracy Gap** compared to MoE-MLP baselines when subjected to input noise (std dev 0.01 - 0.05).
    - While MLPs collapsed to random guessing, MedKAN retained significant predictive power.

2.  **True Interpretability:**
    - Unlike fixed ReLU/GELU activations, our KAN experts learn the exact mathematical function required for the feature.
    - **Visual Proof:** We can plot the exact shape of the learned function for specific PCA components (see below).

3.  **High Accuracy Training:**
    - Trained 31 specialized Experts achieving **99%+ accuracy** on sub-tasks (as seen in training logs).

## 📊 Visualizing the "Glass Box"

### The Learned Function (Expert 0, Feature 18)
Instead of a weight matrix, the KAN expert learned a complex, non-linear B-spline function to process input feature 18. This reveals *exactly* how the model reacts to changes in input intensity.

![Learned Function](https://raw.githubusercontent.com/Saurabsingh778/MedKAN-MoE/main/results/spline_shape_expert_0_feat_18.png)

### Feature Importance
The architecture naturally provides feature attribution scores, allowing doctors/researchers to see which inputs drive the diagnosis.

![Feature Importance](https://raw.githubusercontent.com/Saurabsingh778/MedKAN-MoE/main/results/feature_importance_expert_0.png)

## 🛠️ Architecture

- **Router:** A lightweight Gating Network that routes input samples to the most relevant Experts.
- **Experts:** `K` independent Kolmogorov-Arnold Networks (KANs).
- **KAN Layers:** replace linear weights with learnable 1D functions parametrized as splines.

## 💻 Installation & Usage

```bash
git clone https://github.com/Saurabsingh778/MedKAN-MoE.git
cd MedKAN-MoE
pip install -r requirements.txt