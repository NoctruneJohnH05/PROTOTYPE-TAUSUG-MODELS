import torch

cp = torch.load('autocomplete/static/autocomplete/GRU.pt', map_location='cpu')
sd = cp['model_state']

print("Model Architecture from Checkpoint:")
print("=" * 60)
print(f"fc.weight: {sd['fc.weight'].shape}")
print(f"fc.bias: {sd['fc.bias'].shape}")
print(f"output_layer_norm.weight: {sd['output_layer_norm.weight'].shape}")
print(f"attention.attn.weight: {sd['attention.attn.weight'].shape}")
print(f"gru.weight_ih_l0: {sd['gru.weight_ih_l0'].shape}")
print(f"gru.weight_hh_l0: {sd['gru.weight_hh_l0'].shape}")

print("\nAnalysis:")
print("=" * 60)
fc_in = sd['fc.weight'].shape[1]
print(f"FC input size: {fc_in}")
print(f"GRU is bidirectional: {fc_in == 600}")
print(f"Hidden size: {fc_in // 2 if fc_in == 600 else fc_in}")
