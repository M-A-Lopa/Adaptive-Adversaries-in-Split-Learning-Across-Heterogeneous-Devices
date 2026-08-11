import sys

print("=" * 55)
print("   ENVIRONMENT VERIFICATION")
print("=" * 55)

python_version = sys.version
print(f"\nPython version  : {python_version}")
assert sys.version_info.major == 3, "FAIL: Need Python 3"
assert sys.version_info.minor == 11, "FAIL: Need Python 3.11"
assert sys.version_info.micro == 9,  "FAIL: Need Python 3.11.9"
print("Python check    : PASS (3.11.9)")

import torch
print(f"\nPyTorch version : {torch.__version__}")
assert torch.__version__ == "2.3.1+cu121" or \
       torch.__version__ == "2.3.1", \
       f"FAIL: Expected 2.3.1, got {torch.__version__}"
print("PyTorch check   : PASS (2.3.1)")

cuda_available = torch.cuda.is_available()
print(f"\nCUDA available  : {cuda_available}")
if cuda_available:
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("CUDA check      : PASS")
else:
    print("CUDA check      : WARNING — no GPU detected, will use CPU")

import torchvision
print(f"\ntorchvision     : {torchvision.__version__}")
assert "0.18.1" in torchvision.__version__, \
       f"FAIL: Expected 0.18.1, got {torchvision.__version__}"
print("torchvision     : PASS (0.18.1)")

import numpy as np
print(f"\nNumPy version   : {np.__version__}")
assert np.__version__ == "1.26.4", \
       f"FAIL: Expected 1.26.4, got {np.__version__}"
print("NumPy check     : PASS (1.26.4)")

import matplotlib
import sklearn
import pandas as pd
import seaborn
import tqdm
from PIL import Image

print(f"\nmatplotlib      : {matplotlib.__version__}")
print(f"scikit-learn    : {sklearn.__version__}")
print(f"pandas          : {pd.__version__}")
print(f"seaborn         : {seaborn.__version__}")
print(f"tqdm            : {tqdm.__version__}")

print("\n── Quick GPU tensor test ──")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(64, 3, 32, 32).to(device) 
y = torch.randn(64, 32, 8, 8).to(device)  
print(f"Input tensor    : {x.shape} on {device}")
print(f"Smashed tensor  : {y.shape} on {device}")
print("Tensor test     : PASS")

print("\n" + "=" * 55)
print("   ALL CHECKS PASSED — READY TO START CODING")
print("=" * 55)