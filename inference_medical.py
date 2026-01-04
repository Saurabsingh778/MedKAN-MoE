import torch
import torch.nn as nn
import os
import numpy as np
from moe_kan_lib import KANNetwork, MoEKAN

class MedicalKANSystem:
    def __init__(self, model_dir, num_experts=32, input_dim=20, output_dim=19756, device='cuda'):
        self.model_dir = model_dir
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.device = device
        self.output_dim = output_dim
        
        # 1. Load Router (Always stays on GPU)
        # We need to reconstruct the MoE container just to get the router structure
        # or define the router structure manually matching the training code.
        self.router = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        ).to(device)
        
        print("Loading Router...")
        self.router.load_state_dict(torch.load('router_weights.pth', weights_only=True))
        self.router.eval()
        
        # 2. Expert Cache
        # We will load experts into CPU memory to save VRAM, 
        # and only move them to GPU when needed.
        self.expert_cache = {} 
        self.kan_dims = [20, 64, 32] # Must match training

    def _load_expert(self, expert_id):
        """Loads an expert from disk if not in cache"""
        if expert_id in self.expert_cache:
            return self.expert_cache[expert_id]
        
        path = os.path.join(self.model_dir, f'expert_{expert_id}_best.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert {expert_id} model missing!")
            
        checkpoint = torch.load(path, map_location='cpu') # Load to CPU initially
        
        # Rebuild Model
        expert_net = KANNetwork([20] + self.kan_dims[1:], grid_size=5, spline_order=3)
        final_layer = nn.Linear(self.kan_dims[-1], self.output_dim)
        
        expert_net.load_state_dict(checkpoint['expert_state'])
        final_layer.load_state_dict(checkpoint['final_state'])
        
        self.expert_cache[expert_id] = (expert_net, final_layer)
        return expert_net, final_layer

    def predict(self, x_input):
        """
        x_input: Tensor of shape (Batch, 20)
        Returns: Predicted Class IDs
        """
        x_input = x_input.to(self.device)
        batch_size = x_input.size(0)
        
        # 1. Route
        with torch.no_grad():
            router_logits = self.router(x_input)
            expert_assignments = torch.argmax(router_logits, dim=1)
        
        # Prepare output container
        final_predictions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # 2. Process by Expert Group
        # Instead of loop 0..32, only loop through experts needed for this batch
        needed_experts = torch.unique(expert_assignments).tolist()
        
        for exp_id in needed_experts:
            # Find which samples belong to this expert
            mask = (expert_assignments == exp_id)
            sub_input = x_input[mask]
            
            # Load Expert to GPU
            kan, lin = self._load_expert(exp_id)
            kan.to(self.device).eval()
            lin.to(self.device).eval()
            
            with torch.no_grad():
                features = kan(sub_input)
                logits = lin(features)
                preds = torch.argmax(logits, dim=1)
            
            # Store results
            final_predictions[mask] = preds
            
            # Optional: Move back to CPU to save VRAM if batch is huge
            # kan.cpu() 
            # lin.cpu()
            
        return final_predictions

# ==========================================
# DEMO
# ==========================================
if __name__ == "__main__":
    # Load metadata to decode classes
    data_meta = torch.load('kan_deep_data.pt', weights_only=False)
    classes = data_meta['classes']
    
    # Initialize System
    system = MedicalKANSystem(model_dir='trained_models')
    
    # Fake Test Data (Normally you would embed text here)
    # Let's take 5 random vectors from the dataset to be real
    real_inputs = data_meta['inputs'][:5]
    real_labels = data_meta['labels'][:5]
    
    print("\n--- Running Inference ---")
    preds = system.predict(real_inputs)
    
    for i in range(5):
        true_code = classes[real_labels[i]]
        pred_code = classes[preds[i]]
        print(f"Sample {i}: True {true_code:<10} | Pred {pred_code:<10}")