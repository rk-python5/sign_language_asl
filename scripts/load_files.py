import numpy as np
import torch as trc

print("Numpy File")
print(np.load("landmarks_dataset.npz"))
print("PyTorch File")
print(trc.load("sign_language_model.pth", map_location=trc.device("cpu")))
