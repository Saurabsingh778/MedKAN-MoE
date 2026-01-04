import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from moe_kan_lib import KANNetwork

# --- CONFIG ---
EXPERT_ID = 0  # Let's look at Expert 0 (or any other)
MODEL_PATH = f'trained_models/expert_{EXPERT_ID}_best.pth'
INPUT_DIM = 20
HIDDEN_LAYERS = [20, 64, 32]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_feature_importance(expert_id):
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Expert {expert_id} not found.")
        return None

    model = KANNetwork([INPUT_DIM] + HIDDEN_LAYERS[1:], grid_size=5, spline_order=3).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['expert_state'])
    model.eval()

    # 2. Extract Weights from First Layer
    # The first layer connects Input (20) -> Hidden (64)
    # We want to see which Input contributes most to the Hidden layer.
    
    layer1 = model.layers[0] # EfficientKANLinear
    
    # spline_weight shape: (Out, In * (Grid+Order))
    # base_weight shape:   (Out, In)
    
    # We calculate the L1 norm (magnitude) of the weights for each input index
    with torch.no_grad():
        # Analyze Spline Contribution
        spline_w = layer1.spline_weight.abs() # (64, 20 * 8)
        # Reshape to (Out, In, Coeffs)
        spline_w = spline_w.view(64, 20, -1)
        # Sum across Out and Coeffs to get score per Input
        spline_score = spline_w.sum(dim=(0, 2))
        
        # Analyze Base Linear Contribution
        base_w = layer1.base_weight.abs() # (64, 20)
        base_score = base_w.sum(dim=0)
        
        # Total Score
        total_score = spline_score + base_score
        
    return total_score.cpu().numpy()

# --- PLOT ---
scores = get_feature_importance(EXPERT_ID)

if scores is not None:
    # Normalize scores
    scores = scores / scores.max()
    
    plt.figure(figsize=(12, 6))
    x = np.arange(INPUT_DIM)
    
    # Create gradient bar chart
    sns.barplot(x=x, y=scores, palette="viridis")
    
    plt.title(f"KAN Feature Importance (Expert {EXPERT_ID})")
    plt.xlabel("PCA Feature Index (0-19)")
    plt.ylabel("Importance (Activation Magnitude)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Highlight top 3 features
    top_3 = scores.argsort()[-3:][::-1]
    print(f"\nTop 3 Most Important Features for Expert {EXPERT_ID}: {top_3}")
    print("These PCA components hold the key information for this group of diseases.")
    
    plt.savefig(f"feature_importance_expert_{EXPERT_ID}.png")
    plt.show()