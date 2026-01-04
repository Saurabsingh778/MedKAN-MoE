import torch
from moe_kan_lib import KANNetwork
from moe_mlp_lib import MLPNetwork

kan = KANNetwork([20, 64, 32], grid_size=5, spline_order=3)
mlp = MLPNetwork([20, 64, 32])

k_params = sum(p.numel() for p in kan.parameters())
m_params = sum(p.numel() for p in mlp.parameters())

print(f"KAN Parameters: {k_params:,}")
print(f"MLP Parameters: {m_params:,}")
print(f"Ratio: KAN is {k_params/m_params:.1f}x larger")