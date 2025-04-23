import torch

checkpoint = torch.load('latent_top768_analysis.pt')
print(checkpoint.keys())
print(checkpoint['text_samples'])