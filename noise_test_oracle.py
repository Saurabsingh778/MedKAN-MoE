import torch
import torch.nn as nn
import numpy as np
import os
from moe_kan_lib import KANNetwork
from compare_kan_vs_mlp import LargeMLPNetwork, router
from sklearn.metrics import accuracy_score

# CONFIG
TEST_SIZE = 5000
BATCH_SIZE = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use much smaller noise steps
NOISE_LEVELS = [0.0, 0.01, 0.03, 0.05, 0.1] 

def run_oracle_inference(model_type, X_clean, noise_level):
    # Create Noise
    noise = torch.randn_like(X_clean) * noise_level
    X_noisy = X_clean + noise
    
    preds = []
    
    for i in range(0, len(X_clean), BATCH_SIZE):
        batch_clean = X_clean[i : i+BATCH_SIZE].to(DEVICE)
        batch_noisy = X_noisy[i : i+BATCH_SIZE].to(DEVICE)
        
        # --- ORACLE ROUTING ---
        # We use the CLEAN data to decide which expert to use.
        # This isolates the robustness of the KAN/MLP itself.
        with torch.no_grad():
            r_logits = router(batch_clean)
            assignments = torch.argmax(r_logits, dim=1)
        
        batch_preds = torch.zeros(len(batch_clean), dtype=torch.long, device=DEVICE)
        needed = torch.unique(assignments).tolist()
        
        for exp_id in needed:
            mask = (assignments == exp_id)
            sub_x_noisy = batch_noisy[mask] # Feed NOISY data to the expert
            
            # Load Model
            if model_type == 'KAN':
                model = KANNetwork([20, 64, 32], grid_size=5, spline_order=3)
                path = f'trained_models/expert_{exp_id}_best.pth'
            else:
                model = LargeMLPNetwork()
                path = f'trained_models_mlp_large/expert_{exp_id}_best.pth'
            
            final = nn.Linear(32, 19756)
            
            if not os.path.exists(path): continue
            
            ckpt = torch.load(path, map_location='cpu')
            model.load_state_dict(ckpt['expert_state'])
            final.load_state_dict(ckpt['final_state'])
            
            model.to(DEVICE).eval()
            final.to(DEVICE).eval()
            
            with torch.no_grad():
                out = final(model(sub_x_noisy))
                batch_preds[mask] = torch.argmax(out, dim=1)
            
            del model, final
            torch.cuda.empty_cache()
            
        preds.extend(batch_preds.cpu().numpy())
    return np.array(preds)

# MAIN
print("Loading Data...")
data = torch.load('kan_deep_data.pt', weights_only=False)
indices = torch.randperm(len(data['inputs']))[:TEST_SIZE]
X_test = data['inputs'][indices]
y_test = data['labels'][indices].numpy()

print(f"\n--- ORACLE ROBUSTNESS TEST (Routing with Clean Data) ---")
print(f"{'Noise':<10} | {'KAN Acc':<10} | {'MLP Acc':<10} | {'Diff'}")
print("-" * 45)

for noise in NOISE_LEVELS:
    kan_pred = run_oracle_inference('KAN', X_test, noise)
    mlp_pred = run_oracle_inference('MLP', X_test, noise)
    
    acc_k = accuracy_score(y_test, kan_pred)
    acc_m = accuracy_score(y_test, mlp_pred)
    
    print(f"{noise:<10} | {acc_k:.4f}     | {acc_m:.4f}     | +{(acc_k-acc_m)*100:.1f}%")