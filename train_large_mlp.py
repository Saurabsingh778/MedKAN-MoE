import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import numpy as np

# --- CONFIG ---
DATA_DIR = 'expert_data_splits'
MODEL_DIR = 'trained_models_mlp_large' # New folder
NUM_EXPERTS = 32
# We increase depth and width to match KAN parameter count (~30k)
LAYERS = [20, 200, 100, 32] 
OUTPUT_DIM = 19756        
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024
LR = 0.005

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class LargeMLPNetwork(nn.Module):
    def __init__(self):
        super().__init__()
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

def train_large_expert(expert_id):
    save_path = os.path.join(MODEL_DIR, f'expert_{expert_id}_best.pth')
    if os.path.exists(save_path): return

    print(f"🚀 Training Large MLP Expert {expert_id}...")

    # Load Data
    path = os.path.join(DATA_DIR, f'expert_{expert_id}.pt')
    if not os.path.exists(path): return
    data = torch.load(path, weights_only=False)
    if len(data['labels']) == 0: return

    dataset = TensorDataset(data['inputs'], data['labels'])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize
    model = LargeMLPNetwork().to(DEVICE)
    final_layer = nn.Linear(32, OUTPUT_DIM).to(DEVICE)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': final_layer.parameters()}
    ], lr=LR)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(200): # Max epochs
        model.train()
        total_loss = 0
        batches = 0
        
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = final_layer(model(bx))
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            
        avg_loss = total_loss / batches
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({'expert_state': model.state_dict(), 'final_state': final_layer.state_dict()}, save_path)
        else:
            patience_counter += 1
            if patience_counter >= 15: break # Early stop

    del model, final_layer, optimizer, loader, dataset
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 1. Print Parameter Count Check
    dummy = LargeMLPNetwork()
    params = sum(p.numel() for p in dummy.parameters())
    print(f"Large MLP Base Parameters: {params:,} (Targeting ~30k match)")
    
    # 2. Train
    for i in range(NUM_EXPERTS):
        train_large_expert(i)