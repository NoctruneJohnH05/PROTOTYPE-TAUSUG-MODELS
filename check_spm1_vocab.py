import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load('c:/Users/Tong/Desktop/final gru tausug model/tausug_spm1.model')

print(f'tausug_spm1.model vocab size: {sp.GetPieceSize()}')
print(f'\nSample tokens:')
for i in range(10):
    print(f'  {i}: {sp.id_to_piece(i)}')

print(f'\nTokens around 5000:')
for i in range(4995, min(5005, sp.GetPieceSize())):
    print(f'  {i}: {sp.id_to_piece(i)}')

print(f'\nLast 5 tokens:')
for i in range(sp.GetPieceSize()-5, sp.GetPieceSize()):
    print(f'  {i}: {sp.id_to_piece(i)}')

# Test encoding
test_words = ["kalasahan", "miyaotu", "mayta"]
print(f'\nTest encodings:')
for word in test_words:
    tokens = sp.encode(word, out_type=int)
    print(f'  "{word}": {tokens}')
    oov = [t for t in tokens if t >= 5000]
    if oov:
        print(f'    WARNING: {len(oov)} tokens >= 5000: {oov}')
