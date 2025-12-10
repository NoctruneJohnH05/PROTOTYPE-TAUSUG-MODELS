"""
Test script to verify GRU predictions with correct 5000-token vocab
"""
import sys
sys.path.insert(0, 'c:\\Users\\Tong\\Desktop\\final gru tausug model\\PROTOTYPE-TAUSUG-MODELS')

from autocomplete.utils import get_top_3_preds_gru
import sentencepiece as spm

# Test prompts from user's screenshots
test_prompts = [
    "kalasahan",
    "miyaotu",
    "mayta",
    "sin hi",
    "pagkaun"
]

print("=" * 60)
print("GRU PREDICTIONS WITH CORRECT 5000-TOKEN VOCAB")
print("=" * 60)

# Load the new tokenizer to verify vocab size
sp = spm.SentencePieceProcessor()
sp.Load('c:/Users/Tong/Desktop/final gru tausug model/PROTOTYPE-TAUSUG-MODELS/autocomplete/static/autocomplete/tausug_spm1.model')
print(f"\n✅ Using tausug_spm1.model (vocab size: {sp.GetPieceSize()})")
print(f"   Model vocab: 5000")
print(f"   Tokenizer vocab: {sp.GetPieceSize()}")
print(f"   Vocab match: {'YES ✅' if sp.GetPieceSize() == 5000 else 'NO ❌'}\n")

for prompt in test_prompts:
    print(f"\nInput: '{prompt}'")
    
    # Encode to check for OOV tokens
    tokens = sp.encode(prompt.lower().strip(), out_type=int)
    print(f"  Tokens: {tokens}")
    
    # Check if any tokens are out of range
    oov_count = sum(1 for t in tokens if t >= 5000)
    if oov_count > 0:
        print(f"  ⚠️  WARNING: {oov_count} out-of-vocab tokens!")
    else:
        print(f"  ✅ All tokens in vocab range")
    
    # Get predictions
    predictions = get_top_3_preds_gru(
        prompt,
        sp="autocomplete/tausug_spm1.model",
        model_path="autocomplete/GRU-CHECKPOINT.pt"
    )
    
    print(f"  Predictions: {predictions}")
    print("-" * 60)

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
