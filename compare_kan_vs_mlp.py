import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from moe_kan_lib import KANNetwork
import os

# --- CONFIG ---
TEST_SIZE = 5000 
BATCH_SIZE = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directories
KAN_DIR = 'trained_models'
MLP_DIR = 'trained_models_mlp_large' # Pointing to the Large MLP folder

# Architecture Dims (For KAN)
KAN_LAYERS = [20, 64, 32]
OUTPUT_DIM = 19756

# ==========================================
# 1. DEFINE LARGE MLP CLASS (Must match training script)
# ==========================================
class LargeMLPNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches the structure used in train_large_mlp.py
        self.net = nn.Sequential(
            nn.Linear(20, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 32),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. LOAD ROUTER
# ==========================================
# Router (Shared by both)
router = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32)
).to(DEVICE)

if os.path.exists('router_weights.pth'):
    router.load_state_dict(torch.load('router_weights.pth', weights_only=True))
    router.eval()
else:
    print("❌ Critical: router_weights.pth missing!")
    exit()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def load_expert(model_type, expert_id):
    if model_type == 'KAN':
        # Load KAN
        model = KANNetwork([20] + KAN_LAYERS[1:], grid_size=5, spline_order=3)
        path = os.path.join(KAN_DIR, f'expert_{expert_id}_best.pth')
    else:
        # Load Large MLP
        model = LargeMLPNetwork()
        path = os.path.join(MLP_DIR, f'expert_{expert_id}_best.pth')
        
    final = nn.Linear(32, OUTPUT_DIM) # Both end with 32 -> Output
    
    if not os.path.exists(path): 
        # print(f"Warning: Model {path} not found")
        return None, None
    
    try:
        ckpt = torch.load(path, map_location='cpu')
        model.load_state_dict(ckpt['expert_state'])
        final.load_state_dict(ckpt['final_state'])
    except RuntimeError as e:
        print(f"❌ Error loading expert {expert_id} ({model_type}): {e}")
        return None, None
        
    return model, final

def run_inference(model_type, X_test):
    preds = []
    
    # Batch processing
    for i in range(0, len(X_test), BATCH_SIZE):
        batch_x = X_test[i : i+BATCH_SIZE].to(DEVICE)
        
        # Route
        with torch.no_grad():
            r_logits = router(batch_x)
            assignments = torch.argmax(r_logits, dim=1)
        
        batch_preds = torch.zeros(len(batch_x), dtype=torch.long, device=DEVICE)
        
        needed = torch.unique(assignments).tolist()
        for exp_id in needed:
            mask = (assignments == exp_id)
            sub_x = batch_x[mask]
            
            # Load Expert
            net, final = load_expert(model_type, exp_id)
            if net is None: continue 
            
            net.to(DEVICE).eval()
            final.to(DEVICE).eval()
            
            with torch.no_grad():
                # MLP vs KAN forward pass
                features = net(sub_x)
                out = final(features)
                batch_preds[mask] = torch.argmax(out, dim=1)
            
            # Free VRAM immediately
            del net, final
            torch.cuda.empty_cache()
            
        preds.extend(batch_preds.cpu().numpy())
    return np.array(preds)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Loading Test Data...")
    data = torch.load('kan_deep_data.pt', weights_only=False)
    
    # Use fixed seed for fair comparison
    torch.manual_seed(42) 
    indices = torch.randperm(len(data['inputs']))[:TEST_SIZE]
    X_test = data['inputs'][indices]
    y_test = data['labels'][indices].numpy()

    print(f"\n--- EVALUATING KAN MOE (30k Params) ---")
    kan_preds = run_inference('KAN', X_test)
    kan_acc = accuracy_score(y_test, kan_preds)
    print(f"KAN Accuracy: {kan_acc:.4f}")

    print(f"\n--- EVALUATING LARGE MLP MOE (30k Params) ---")
    mlp_preds = run_inference('MLP', X_test)
    mlp_acc = accuracy_score(y_test, mlp_preds)
    print(f"MLP Accuracy: {mlp_acc:.4f}")

    print("\n" + "="*30)
    print("FINAL RESEARCH RESULTS")
    print("="*30)
    print(f"KAN: {kan_acc*100:.2f}%")
    print(f"MLP: {mlp_acc*100:.2f}%")
    
    if kan_acc > mlp_acc:
        print(f"🏆 KAN WINS by +{(kan_acc - mlp_acc)*100:.2f}%")
        print("Conclusion: KANs are architecturally superior for this task, regardless of parameter count.")
    else:
        print(f"🏆 MLP WINS by +{(mlp_acc - kan_acc)*100:.2f}%")