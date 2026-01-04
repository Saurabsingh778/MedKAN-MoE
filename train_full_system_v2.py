import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
import time
import numpy as np
import gc # Garbage collector for memory leaks
from moe_kan_lib import KANNetwork, MoEKAN

# --- CONFIGURATION ---
DATA_DIR = 'expert_data_splits'
MODEL_DIR = 'trained_models'
LOG_FILE = 'training_log.txt'
NUM_EXPERTS = 32
KAN_LAYERS = [20, 64, 32] 
OUTPUT_DIM = 19756        
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training Hyperparameters
MAX_EPOCHS = 200          
PATIENCE = 15             
BATCH_SIZE = 1024
LR = 0.005

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ==============================================================================
# CLASS: DUAL LOGGER (Console + File)
# ==============================================================================
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')  # "a" for append

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure it writes immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect print to our logger
sys.stdout = DualLogger(LOG_FILE)

# ==============================================================================
# HELPER: TRAIN SINGLE EXPERT
# ==============================================================================
def train_expert(expert_id):
    save_path = os.path.join(MODEL_DIR, f'expert_{expert_id}_best.pth')
    
    # --- RESUME LOGIC ---
    # If the model already exists, we skip training. 
    # This solves your crash issue: just re-run the script, and it picks up at 21.
    if os.path.exists(save_path):
        print(f"✅ Expert {expert_id} already trained. Skipping...")
        return

    print(f"\n" + "="*40)
    print(f"🚀 STARTING EXPERT {expert_id}/{NUM_EXPERTS-1}")
    print("="*40)

    # 1. Load Data
    path = os.path.join(DATA_DIR, f'expert_{expert_id}.pt')
    if not os.path.exists(path):
        print(f"⚠️ Data file not found for expert {expert_id}. Skipping.")
        return

    data = torch.load(path, weights_only=False)
    inputs = data['inputs']
    labels = data['labels']
    
    if len(labels) == 0:
        print("Expert has no data. Skipping.")
        return

    # Create Loader
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model
    model = KANNetwork(
        [20] + KAN_LAYERS[1:], 
        grid_size=5, 
        spline_order=3
    ).to(DEVICE)
    
    final_layer = nn.Linear(KAN_LAYERS[-1], OUTPUT_DIM).to(DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': final_layer.parameters()}
    ], lr=LR)
    
    # Fixed: Removed 'verbose=True' to prevent error
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        total_acc = 0
        batches = 0
        
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            
            optimizer.zero_grad()
            feat = model(bx)
            logits = final_layer(feat)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
            acc = (torch.argmax(logits, dim=1) == by).float().mean().item()
            total_loss += loss.item()
            total_acc += acc
            batches += 1
            
        avg_loss = total_loss / batches
        avg_acc = total_acc / batches
        
        # Step Scheduler
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch+1) % 5 == 0:
            print(f"   Ep {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | LR: {current_lr:.5f}")

        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            state = {
                'expert_state': model.state_dict(),
                'final_state': final_layer.state_dict(),
                'stats': {'acc': avg_acc, 'loss': avg_loss}
            }
            torch.save(state, save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"   🛑 Early stopping at epoch {epoch+1}. Best Loss: {best_loss:.4f}")
                break

    # 4. Aggressive Cleanup (Crucial for preventing CUDA Unknown Error)
    del model, final_layer, optimizer, loader, dataset, inputs, labels, scheduler, criterion
    torch.cuda.empty_cache()
    gc.collect() # Force Python garbage collection

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print(f"Logging started at {time.ctime()}")
    print(f"Saving logs to: {LOG_FILE}")
    start_time = time.time()
    
    if not os.path.exists('router_weights.pth'):
        print("❌ CRITICAL: 'router_weights.pth' not found!")
        exit()

    for i in range(NUM_EXPERTS):
        try:
            train_expert(i)
        except Exception as e:
            # Check for Memory Error specifically
            if "CUDA" in str(e) or "memory" in str(e):
                print(f"\n❌ CRITICAL GPU ERROR on Expert {i}: {e}")
                print("⚠️ Recommendation: Restart the script. It will auto-resume from this expert.")
                sys.exit(1) # Stop script so you can restart cleanly
            else:
                print(f"❌ Error training Expert {i}: {e}")
            
    total_time = (time.time() - start_time) / 3600
    print(f"\n🎉 ALL DONE! Total time: {total_time:.2f} hours.")