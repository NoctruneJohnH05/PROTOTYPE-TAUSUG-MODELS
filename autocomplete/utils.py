import sentencepiece as spm
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import keras
from pathlib import Path
import tensorflow as tf
from keras.saving import register_keras_serializable
import torch
import torch.nn as nn
import difflib

START_TOKEN_ID = 1
END_TOKEN_ID = 2
MAX_LEN = 426
PAD_TOKEN = 0


# ==================== Model Cache Singleton ====================
class ModelCache:
    """
    Singleton class to cache loaded models and SentencePiece processors
    to avoid repeated file I/O and model loading on each request.
    """
    _instance = None
    _models = {}
    _sp_processors = {}
    _vocabulary = None
    _fuzzy_match_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def get_model(self, model_type, model_path):
        """
        Get a cached model or load it if not cached.
        
        Args:
            model_type: Type of model ('lstm', 'bidirectional', 'gru')
            model_path: Path to the model file
        
        Returns:
            Loaded model (and config for GRU models)
        """
        cache_key = f"{model_type}:{model_path}"
        
        if cache_key not in self._models:
            print(f"[MODEL CACHE] Loading {model_type} model for the first time: {model_path}")
            
            if model_type == 'gru':
                model, config = load_gru_model(model_path)
                self._models[cache_key] = (model, config)
            else:
                model = load_rrn_model(model_path)
                self._models[cache_key] = model
            
            print(f"[MODEL CACHE] {model_type} model cached successfully")
        else:
            print(f"[MODEL CACHE] Using cached {model_type} model")
        
        return self._models[cache_key]
    
    def get_sp_processor(self, sp_path):
        """
        Get a cached SentencePiece processor or load it if not cached.
        
        Args:
            sp_path: Path to the SentencePiece model file
        
        Returns:
            Loaded SentencePiece processor
        """
        if sp_path not in self._sp_processors:
            print(f"[SP CACHE] Loading SentencePiece model for the first time: {sp_path}")
            self._sp_processors[sp_path] = load_spm_model(sp_path)
            print(f"[SP CACHE] SentencePiece model cached successfully")
        else:
            print(f"[SP CACHE] Using cached SentencePiece processor")
        
        return self._sp_processors[sp_path]
    
    def get_vocabulary(self):
        """
        Get cached Tausug vocabulary set or build it from corpus.
        
        Returns:
            Set of valid Tausug words
        """
        if self._vocabulary is None:
            print("[VOCAB CACHE] Building vocabulary from corpus...")
            corpus_path = Path(__file__).resolve().parent.joinpath(
                'static', 'autocomplete', 'cleaned_tausug_corpus_augmented.txt'
            )
            
            try:
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    words = set()
                    for line in f:
                        words.update(line.strip().split())
                    self._vocabulary = words
                print(f"[VOCAB CACHE] Vocabulary built: {len(self._vocabulary)} unique words")
            except Exception as e:
                print(f"[VOCAB CACHE] Error loading vocabulary: {e}")
                self._vocabulary = set()
        
        return self._vocabulary
    
    def get_fuzzy_match(self, word, cutoff=0.6):
        """
        Get cached fuzzy match or compute it.
        
        Args:
            word: Word to find match for
            cutoff: Similarity threshold (0.0 to 1.0)
        
        Returns:
            Best matching word from vocabulary or original word
        """
        cache_key = f"{word}:{cutoff}"
        
        if cache_key not in self._fuzzy_match_cache:
            vocab = self.get_vocabulary()
            matches = difflib.get_close_matches(word, vocab, n=1, cutoff=cutoff)
            self._fuzzy_match_cache[cache_key] = matches[0] if matches else word
            
            # Limit cache size
            if len(self._fuzzy_match_cache) > 1000:
                # Remove oldest 100 entries
                for _ in range(100):
                    self._fuzzy_match_cache.pop(next(iter(self._fuzzy_match_cache)))
        
        return self._fuzzy_match_cache[cache_key]
    
    def get_prefix_matches(self, prefix, max_results=5):
        """
        Get words from vocabulary that start with the given prefix.
        Fast autocomplete functionality.
        
        Args:
            prefix: Word prefix to match
            max_results: Maximum number of matches to return
        
        Returns:
            List of matching words
        """
        if not prefix:
            return []
        
        vocab = self.get_vocabulary()
        matches = [w for w in vocab if w.startswith(prefix.lower())]
        return sorted(matches)[:max_results]
    
    def preload_all_models(self):
        """
        Preload all models on server startup for faster first requests.
        """
        print("\n" + "="*60)
        print("PRELOADING ALL MODELS INTO MEMORY...")
        print("="*60)
        
        try:
            # Preload LSTM model
            self.get_model('lstm', 'autocomplete/LSTM-TESTING.keras')
            self.get_sp_processor('autocomplete/tausug_spm.model')
            
            # Preload Bidirectional model
            self.get_model('bidirectional', 'autocomplete/BIDIRECTIONAL-FINETUNED2.keras')
            
            # Preload GRU model
            self.get_model('gru', 'autocomplete/GRU.pt')
            self.get_sp_processor('autocomplete/gru_spm.model')
            
            # Preload vocabulary
            self.get_vocabulary()
            
            print("="*60)
            print("ALL MODELS PRELOADED SUCCESSFULLY!")
            print("="*60 + "\n")
        except Exception as e:
            print(f"[WARNING] Error preloading models: {e}")
            print("Models will be loaded on first request instead.\n")


# Global singleton instance
model_cache = ModelCache()


# ==================== Keras/TensorFlow Functions ====================
@register_keras_serializable(package="Custom", name="masked_loss")
def masked_loss(y_true, y_pred):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction='none'
    )
    loss = loss_fn(y_true, y_pred)

    ignore_mask = tf.logical_and(
        tf.not_equal(y_true, PAD_TOKEN),
        tf.logical_and(
            tf.not_equal(y_true, START_TOKEN_ID),
            tf.not_equal(y_true, END_TOKEN_ID)
        )
    )
    mask = tf.cast(ignore_mask, tf.float32)

    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

@register_keras_serializable(package="Custom", name="masked_accuracy")
def masked_accuracy(y_true, y_pred):
    y_pred_ids = tf.argmax(y_pred, axis=-1)
    y_pred_ids = tf.cast(y_pred_ids, y_true.dtype)

    matches = tf.cast(tf.equal(y_true, y_pred_ids), tf.float32)

    ignore_mask = tf.logical_and(
        tf.not_equal(y_true, PAD_TOKEN),
        tf.logical_and(
            tf.not_equal(y_true, START_TOKEN_ID),
            tf.not_equal(y_true, END_TOKEN_ID)
        )
    )
    mask = tf.cast(ignore_mask, tf.float32)

    matches *= mask

    return tf.reduce_sum(matches) / tf.reduce_sum(mask)


def load_rrn_model(model):
    model_path = Path(model)
    if not model_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        candidate = project_root.joinpath(model_path)
        if candidate.exists():
            model_path = candidate
        else:
            candidate2 = Path(__file__).resolve().parent.joinpath('static', 'autocomplete', model_path.name)
            candidate3 = project_root.joinpath('autocomplete', 'static', 'autocomplete', model_path.name)
            if candidate2.exists():
                model_path = candidate2
            elif candidate3.exists():
                model_path = candidate3
            else:
                raise ValueError(f"Model file not found: filepath={model}. Checked: {candidate}, {candidate2}, {candidate3}")

    return keras.models.load_model(
        str(model_path), 
        custom_objects={
        "masked_loss": masked_loss,
        "masked_accuracy": masked_accuracy
    })

def load_spm_model(model):
    model_path = Path(model)
    if not model_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        candidate = project_root.joinpath(model_path)
        if candidate.exists():
            model_path = candidate
        else:
            candidate2 = Path(__file__).resolve().parent.joinpath('static', 'autocomplete', model_path.name)
            candidate3 = project_root.joinpath('autocomplete', 'static', 'autocomplete', model_path.name)
            if candidate2.exists():
                model_path = candidate2
            elif candidate3.exists():
                model_path = candidate3
            else:
                raise ValueError(f"SentencePiece model file not found: filepath={model}. Checked: {candidate}, {candidate2}, {candidate3}")

    sp_proc = spm.SentencePieceProcessor()
    # Use the binding's Load method
    sp_proc.Load(str(model_path))
    return sp_proc


# ==================== PyTorch GRU Model ====================
class LuongAttention(nn.Module):
    """Luong Attention Mechanism (Multiplicative/Dot-Product)"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        Args:
            hidden: Current hidden state [batch_size, hidden_size]
            encoder_outputs: All previous outputs [batch_size, seq_len, hidden_size]
            mask: Attention mask [batch_size, seq_len]
        """
        query = self.attn(hidden).unsqueeze(1)  # [batch, 1, hidden]
        scores = torch.bmm(query, encoder_outputs.transpose(1, 2))  # [batch, 1, seq_len]
        scores = scores.squeeze(1)  # [batch, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)  # [batch, seq_len]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden]
        context = context.squeeze(1)  # [batch, hidden]
        
        return context, attention_weights


class GRUModel(nn.Module):
    """
    Improved GRU model with Luong Attention and Layer Normalization
    Matches the architecture from the training code
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.25):
        super(GRUModel, self).__init__()
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.25):
        super(GRUModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_TOKEN)
        
        # Layer Normalization for embeddings
        self.embed_layer_norm = nn.LayerNorm(embed_size)
        
        # GRU layer (unidirectional)
        self.gru = nn.GRU(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout,
            bidirectional=False
        )
        
        # Luong Attention Mechanism
        self.attention = LuongAttention(hidden_size)
        
        # Layer Normalization for output
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        
        # Output layer (projects hidden + context to vocabulary)
        # *2 because we concatenate GRU output with attention context
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass with Luong Attention
        
        Args:
            x: Input token IDs [batch, seq_len]
            mask: Attention mask [batch, seq_len]
        
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.size()
        
        # Embedding with normalization
        emb = self.embedding(x)  # [batch, seq_len, embed]
        emb = self.embed_layer_norm(emb)
        
        # GRU forward pass
        gru_out, hidden = self.gru(emb)  # gru_out: [batch, seq_len, hidden]
        gru_out = self.dropout(gru_out)
        
        # Apply Luong Attention at each timestep
        context_vectors = []
        for t in range(seq_len):
            current_hidden = gru_out[:, t, :]  # [batch, hidden]
            
            if t == 0:
                # First token: no context
                context = torch.zeros_like(current_hidden)
            else:
                # Use attention over all previous tokens
                encoder_outputs = gru_out[:, :t, :]  # [batch, t, hidden]
                attn_mask = mask[:, :t] if mask is not None else None
                context, _ = self.attention(current_hidden, encoder_outputs, attn_mask)
            
            context_vectors.append(context)
        
        # Stack context vectors
        context_all = torch.stack(context_vectors, dim=1)  # [batch, seq_len, hidden]
        
        # Apply layer norm to GRU output (not context)
        gru_out_normed = self.output_layer_norm(gru_out)
        
        # Concatenate GRU output with attention context
        combined = torch.cat([gru_out_normed, context_all], dim=-1)  # [batch, seq_len, hidden*2]
        
        # Project to vocabulary
        logits = self.fc(combined)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


def load_gru_model(model_path):
    """
    Load PyTorch GRU model from checkpoint
    """
    model_path = Path(model_path)
    if not model_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        candidate = project_root.joinpath(model_path)
        if candidate.exists():
            model_path = candidate
        else:
            candidate2 = Path(__file__).resolve().parent.joinpath('static', 'autocomplete', model_path.name)
            candidate3 = project_root.joinpath('autocomplete', 'static', 'autocomplete', model_path.name)
            if candidate2.exists():
                model_path = candidate2
            elif candidate3.exists():
                model_path = candidate3
            else:
                raise ValueError(f"GRU model file not found: filepath={model_path}. Checked: {candidate}, {candidate2}, {candidate3}")
    
    # Load checkpoint
    checkpoint = torch.load(str(model_path), map_location='cpu')
    config = checkpoint['config']
    
    # Initialize model
    model = GRUModel(
        vocab_size=config['vocab_size'],
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config.get('dropout', 0.25)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state'])
    model.eval()  # Set to evaluation mode
    
    return model, config


# ==================== Helper Functions ====================
def correct_predictions_with_vocab(predictions, sp_proc):
    """
    Correct predicted words using vocabulary fuzzy matching.
    
    Args:
        predictions: List of predicted words
        sp_proc: SentencePiece processor (for fallback)
    
    Returns:
        List of corrected words
    """
    vocab = model_cache.get_vocabulary()
    corrected = []
    
    for word in predictions:
        word = word.strip().lower()
        if not word:
            continue
            
        # If word is in vocabulary, use it as-is
        if word in vocab:
            corrected.append(word)
        else:
            # Try fuzzy matching
            match = model_cache.get_fuzzy_match(word, cutoff=0.6)
            corrected.append(match)
    
    return corrected


def filter_special_tokens(probs, exclude_tokens=None):
    """
    Set probability to 0 for special tokens (PAD, START, END).
    
    Args:
        probs: Probability array
        exclude_tokens: Set of token IDs to exclude
    
    Returns:
        Filtered probability array
    """
    if exclude_tokens is None:
        exclude_tokens = {PAD_TOKEN, START_TOKEN_ID, END_TOKEN_ID}
    
    probs = probs.copy()
    for token_id in exclude_tokens:
        if token_id < len(probs):
            probs[token_id] = 0
    
    # Renormalize probabilities
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    
    return probs


# ==================== LSTM Prediction Function ====================
def get_top_3_preds(prompt, sp="autocomplete/tausug_spm.model", model_path="autocomplete/LSTM-TESTING.keras", max_len=MAX_LEN, max_generate=1):
    """
    Get top-3 next word predictions using LSTM model with diverse seeding and vocabulary correction.
    
    Args:
        prompt: Input text
        sp: SentencePiece model path
        model_path: LSTM model checkpoint path
        max_len: Maximum sequence length
        max_generate: Number of tokens to generate per sequence (default 1 for next-word prediction)
    """
    # Check for prefix-based autocomplete (instant response)
    words = prompt.strip().split()
    if words:
        last_word = words[-1].lower()
        vocab = model_cache.get_vocabulary()
        
        # If last word is incomplete (not in vocab), try autocomplete
        if last_word and last_word not in vocab:
            prefix_matches = model_cache.get_prefix_matches(last_word, max_results=3)
            if prefix_matches:
                print(f"[AUTOCOMPLETE] Prefix '{last_word}' -> {prefix_matches[:3]}")
                return " ".join(prefix_matches[:3])
            else:
                # No matches found - likely non-Tausug word
                print(f"[WARNING] No vocabulary matches for '{last_word}' - possible non-Tausug word")
                return "[Non-Tausug word detected]"
    
    # Use cached model instead of loading every time
    model = model_cache.get_model('lstm', model_path)
    
    if isinstance(sp, str):
        sp_proc = model_cache.get_sp_processor(sp)
    else:
        sp_proc = sp 

    input_tokens = sp_proc.encode(prompt)
    encoder_input = pad_sequences([input_tokens], maxlen=max_len, padding="post")
    seq = [START_TOKEN_ID] + input_tokens

    # Get probabilities for first step
    decoder_input = np.array([seq])
    preds = model.predict([encoder_input, decoder_input], verbose=0)
    probs = preds[0, -1, :].copy()

    # Filter out special tokens
    probs = filter_special_tokens(probs)
    
    # Get top 3 starting tokens for diverse seeding
    top_3_start = np.argsort(probs)[-3:][::-1]
    
    all_predictions = []
    
    # Generate 3 separate sequences, each starting with a different high-probability token
    for start_token in top_3_start:
        start_token = int(start_token)
        gen_seq = seq + [start_token]
        
        # Continue greedy decoding for remaining tokens (if max_generate > 1)
        for _ in range(max_generate - 1):
            decoder_input = np.array([gen_seq])
            preds = model.predict([encoder_input, decoder_input], verbose=0)
            step_probs = preds[0, -1, :].copy()
            step_probs = filter_special_tokens(step_probs)
            
            # Pick highest probability token
            next_token = int(np.argmax(step_probs))
            gen_seq.append(next_token)
        
        # Extract generated tokens
        gen_tokens = gen_seq[1 + len(input_tokens): 1 + len(input_tokens) + max_generate]
        
        # Decode
        if gen_tokens:
            text = sp_proc.decode(gen_tokens)
            if text.strip():
                all_predictions.append(text.strip())
    
    # Apply vocabulary correction to all predictions
    corrected = correct_predictions_with_vocab(all_predictions, sp_proc)
    
    # Return space-separated predictions
    return " ".join(corrected[:3])



# ==================== Bidirectional Prediction Function ====================
def get_top_3_preds_bidirectional(prompt, sp="autocomplete/tausug_spm.model", model_path="autocomplete/BIDIRECTIONAL-FINETUNED2.keras", max_len=MAX_LEN, max_generate=1):
    """
    Get top-3 next word predictions using Bidirectional model with diverse seeding and vocabulary correction.
    
    Args:
        prompt: Input text
        sp: SentencePiece model path
        model_path: Bidirectional model checkpoint path
        max_len: Maximum sequence length
        max_generate: Number of tokens to generate per sequence (default 1 for next-word prediction)
    """
    # Check for prefix-based autocomplete (instant response)
    words = prompt.strip().split()
    if words:
        last_word = words[-1].lower()
        vocab = model_cache.get_vocabulary()
        
        # If last word is incomplete (not in vocab), try autocomplete
        if last_word and last_word not in vocab:
            prefix_matches = model_cache.get_prefix_matches(last_word, max_results=3)
            if prefix_matches:
                print(f"[AUTOCOMPLETE] Prefix '{last_word}' -> {prefix_matches[:3]}")
                return " ".join(prefix_matches[:3])
            else:
                # No matches found - likely non-Tausug word
                print(f"[WARNING] No vocabulary matches for '{last_word}' - possible non-Tausug word")
                return "[Non-Tausug word detected]"
    
    # Use cached model instead of loading every time
    model = model_cache.get_model('bidirectional', model_path)
    
    if isinstance(sp, str):
        sp_proc = model_cache.get_sp_processor(sp)
    else:
        sp_proc = sp 

    input_tokens = sp_proc.encode(prompt)
    encoder_input = pad_sequences([input_tokens], maxlen=max_len, padding="post")
    seq = [START_TOKEN_ID] + input_tokens

    # Get probabilities for first step
    decoder_input = np.array([seq])
    preds = model.predict([encoder_input, decoder_input], verbose=0)
    probs = preds[0, -1, :].copy()

    # Filter out special tokens
    probs = filter_special_tokens(probs)
    
    # Get top 3 starting tokens for diverse seeding
    top_3_start = np.argsort(probs)[-3:][::-1]
    
    all_predictions = []
    
    # Generate 3 separate sequences, each starting with a different high-probability token
    for start_token in top_3_start:
        start_token = int(start_token)
        gen_seq = seq + [start_token]
        
        # Continue greedy decoding for remaining tokens (if max_generate > 1)
        for _ in range(max_generate - 1):
            decoder_input = np.array([gen_seq])
            preds = model.predict([encoder_input, decoder_input], verbose=0)
            step_probs = preds[0, -1, :].copy()
            step_probs = filter_special_tokens(step_probs)
            
            # Pick highest probability token
            next_token = int(np.argmax(step_probs))
            gen_seq.append(next_token)
        
        # Extract generated tokens
        gen_tokens = gen_seq[1 + len(input_tokens): 1 + len(input_tokens) + max_generate]
        
        # Decode
        if gen_tokens:
            text = sp_proc.decode(gen_tokens)
            if text.strip():
                all_predictions.append(text.strip())
    
    # Apply vocabulary correction to all predictions
    corrected = correct_predictions_with_vocab(all_predictions, sp_proc)
    
    # Return space-separated predictions
    return " ".join(corrected[:3])


# ==================== GRU Prediction Function ====================
def get_top_3_preds_gru(prompt, sp="autocomplete/gru_spm.model", model_path="autocomplete/GRU.pt", max_generate=3):
    """
    Get top-3 next word predictions using PyTorch GRU model with vocabulary correction.
    
    Args:
        prompt: Input text
        sp: SentencePiece model path (uses gru_spm.model with 5000 vocab by default)
        model_path: GRU model checkpoint path
        max_generate: Number of predictions
    """
    # Check for prefix-based autocomplete (instant response)
    words = prompt.strip().split()
    if words:
        last_word = words[-1].lower()
        vocab = model_cache.get_vocabulary()
        
        # If last word is incomplete (not in vocab), try autocomplete
        if last_word and last_word not in vocab:
            prefix_matches = model_cache.get_prefix_matches(last_word, max_results=3)
            if prefix_matches:
                print(f"[AUTOCOMPLETE] Prefix '{last_word}' -> {prefix_matches[:3]}")
                return " ".join(prefix_matches[:3])
            else:
                # No matches found - likely non-Tausug word
                print(f"[WARNING] No vocabulary matches for '{last_word}' - possible non-Tausug word")
                return "[Non-Tausug word detected]"
    
    # Use cached model instead of loading every time
    model, config = model_cache.get_model('gru', model_path)
    
    if isinstance(sp, str):
        sp_proc = model_cache.get_sp_processor(sp)
    else:
        sp_proc = sp
    
    # Encode input text
    input_tokens = sp_proc.encode(prompt.lower().strip())
    
    # ⚠️ CRITICAL: Clip token IDs to model's vocab size
    # GRU model has vocab_size=5000, but SPM might have 7000
    vocab_size = config['vocab_size']
    input_tokens = [min(t, vocab_size - 1) for t in input_tokens]
    
    # If all tokens are out of range, use a safe default
    if not input_tokens:
        input_tokens = [0]  # PAD token
    
    # Convert to tensor
    input_tensor = torch.tensor([input_tokens], dtype=torch.long)
    
    # Get predictions
    with torch.no_grad():
        logits = model(input_tensor)  # Returns [batch, seq_len, vocab_size]
        # Get last token's predictions
        last_logits = logits[0, -1, :]  # [vocab_size]
        probs = torch.softmax(last_logits, dim=-1).numpy()
    
    # Filter out special tokens
    probs = filter_special_tokens(probs)
    
    # Get top 3 token IDs after filtering
    top_3_indices = np.argsort(probs)[-3:][::-1]
    
    # Decode predictions
    predictions = []
    for token_id in top_3_indices:
        word = sp_proc.decode([int(token_id)])
        if word.strip():  # Filter out empty tokens
            predictions.append(word.strip())
    
    # Apply vocabulary correction
    corrected = correct_predictions_with_vocab(predictions, sp_proc)
    
    # Join top predictions
    text = " ".join(corrected[:max_generate])
    
    return text
