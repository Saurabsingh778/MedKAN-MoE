import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from inference_medical import MedicalKANSystem
from tqdm import tqdm

# --- CONFIG ---
TEST_SIZE = 5000  # Number of samples to test
BATCH_SIZE = 1000 # Inference batch size

# Load Data
print("Loading data...")
data = torch.load('kan_deep_data.pt', weights_only=False)
X_all = data['inputs']
y_all = data['labels']
classes = data['classes']

# Pick random test set
indices = torch.randperm(len(X_all))[:TEST_SIZE]
X_test = X_all[indices]
y_test = y_all[indices]

# Load System
system = MedicalKANSystem(model_dir='trained_models')

# Run Inference in Batches
all_preds = []
print(f"Running inference on {TEST_SIZE} samples...")

for i in tqdm(range(0, len(X_test), BATCH_SIZE)):
    batch_x = X_test[i : i+BATCH_SIZE]
    
    # Predict
    preds = system.predict(batch_x)
    all_preds.extend(preds.cpu().numpy())

# Calculate Metrics
y_true = y_test.cpu().numpy()
y_pred = np.array(all_preds)

print("\n" + "="*30)
print("FINAL RESULTS")
print("="*30)
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

# Weighted F1 is better for imbalanced medical data
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"Weighted F1-Score: {f1:.4f}")

# Save detailed report
report = classification_report(y_true, y_pred, target_names=classes[np.unique(np.concatenate((y_true, y_pred)))], output_dict=True)
import json
with open("classification_report.json", "w") as f:
    json.dump(report, f, indent=4)
print("Saved detailed report to classification_report.json")