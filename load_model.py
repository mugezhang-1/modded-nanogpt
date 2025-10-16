import torch
from train_gpt_medium import GPT

# Load checkpoint
checkpoint = torch.load('logs/000_{uuid}/state_step001669.pt')

# Recreate model
model = GPT(vocab_size=50000, num_layers=16, num_heads=8, model_dim=1024, max_seq_len=262144)
model.load_state_dict(checkpoint['model'])