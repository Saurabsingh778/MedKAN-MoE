import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from moe_kan_lib import KANNetwork

# --- CONFIG ---
EXPERT_ID = 0
FEATURE_IDX = 18  # This was the highest bar in your previous feature importance plot
MODEL_PATH = f'trained_models/expert_{EXPERT_ID}_best.pth'
HIDDEN_LAYERS = [20, 64, 32]

# Load Model
model = KANNetwork([20] + HIDDEN_LAYERS[1:], grid_size=5, spline_order=3)
if not os.path.exists(MODEL_PATH):
    print("Model not found!")
    exit()

ckpt = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(ckpt['expert_state'])
model.eval()

# Generate inputs from -3 to +3 (Standard Deviation range)
x_values = torch.linspace(-3, 3, 200)
input_tensor = torch.zeros(200, 20)
# We vary ONLY Feature 18, keeping others at 0 (mean)
input_tensor[:, FEATURE_IDX] = x_values

# Hook into the first layer
layer = model.layers[0]

with torch.no_grad():
    # Pass through the B-Spline layer
    activations = layer(input_tensor)

# Average the output to see the "Average Learned Effect" of this feature
y_values = activations.mean(dim=1).numpy()
x_plot = x_values.numpy()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_values, linewidth=4, color='#FF5733', label='KAN Learned Function')
plt.title(f"What Expert {EXPERT_ID} learned about Feature {FEATURE_IDX}", fontsize=14)
plt.xlabel("Input Value (PCA Component 18)", fontsize=12)
plt.ylabel("Activation Intensity", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig(f"kan_spline_shape_exp{EXPERT_ID}_feat{FEATURE_IDX}.png", dpi=300)
plt.show()

print("Generated spline plot. If this line is CURVED, you have proof of non-linearity.")