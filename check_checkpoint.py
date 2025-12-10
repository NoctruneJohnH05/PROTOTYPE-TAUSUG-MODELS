import torch

checkpoint = torch.load('autocomplete/static/autocomplete/GRU-CHECKPOINT.pt', map_location='cpu')

print('=' * 60)
print('GRU-CHECKPOINT.pt Configuration:')
print('=' * 60)

config = checkpoint.get('config', {})
print('\nConfig:')
for k, v in config.items():
    print(f'  {k}: {v}')

print('\nCheckpoint Keys:')
for k in checkpoint.keys():
    print(f'  - {k}')

print('\nModel State Keys (first 10):')
state_keys = list(checkpoint['model_state'].keys())
for k in state_keys[:10]:
    print(f'  - {k}')

print(f'\nTotal model parameters: {len(state_keys)}')

if 'epoch' in checkpoint:
    print(f'\nTrained for {checkpoint["epoch"]} epochs')
    
if 'val_top5' in checkpoint:
    print(f'Best Val Top-5 Accuracy: {checkpoint["val_top5"]:.2%}')

if 'val_acc' in checkpoint:
    print(f'Best Val Top-1 Accuracy: {checkpoint["val_acc"]:.2%}')

# Check embedding layer size
embedding_weight = checkpoint['model_state'].get('embedding.weight')
if embedding_weight is not None:
    vocab_size, embed_size = embedding_weight.shape
    print(f'\nEmbedding Matrix Shape: [{vocab_size}, {embed_size}]')
    print(f'  -> Actual Vocab Size in Model: {vocab_size}')
    print(f'  -> Embedding Dimension: {embed_size}')
