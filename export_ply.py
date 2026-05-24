import os
import torch
from gaussian_me.model import GaussianModel

# 1) Where your trained npz lives
npz_path = os.path.join("Test_Model_v9",
                        "point_cloud",
                        "iteration_1000",
                        "point_cloud.npz")

# 2) Where you want to write the mesh
out_ply = os.path.join("Test_Model_v9",
                       "iteration_1000.ply")
os.makedirs(os.path.dirname(out_ply), exist_ok=True)

# 3) Load & move to CPU
pc = GaussianModel.from_npz(npz_path)
pc.to(torch.device("cpu"))

# 4) Export
pc.save_ply(out_ply,
            background_color=[0.0, 0.0, 0.0, 1.0])  # RGBA 0–1
print(f"✅ PLY written to {out_ply}")
