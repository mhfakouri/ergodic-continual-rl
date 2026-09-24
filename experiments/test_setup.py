import gymnasium as gym
import stable_baselines3 as sb3
import torch

print("Environment setup OK")
print("Gymnasium:", gym.__version__)
print("Stable-Baselines3:", sb3.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
