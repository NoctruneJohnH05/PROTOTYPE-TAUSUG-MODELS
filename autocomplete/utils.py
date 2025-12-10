import sentencepiece as spm
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import keras
from pathlib import Path
import tensorflow as tf
from keras.saving import register_keras_serializable
import torch
import torch.nn as nn

START_TOKEN_ID = 1
END_TOKEN_ID = 2
MAX_LEN = 426
PAD_TOKEN = 0


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


# ==================== LSTM Prediction Function ====================
def get_top_3_preds(prompt, sp="autocomplete/tausug_spm.model", model_path="autocomplete/LSTM-TESTING.keras", max_len=MAX_LEN, max_generate=3):
    """
    Get top-3 next word predictions using LSTM model
    
    Args:
        prompt: Input text
        sp: SentencePiece model path
        model_path: LSTM model checkpoint path
        max_len: Maximum sequence length
        max_generate: Number of predictions
    """
    model = load_rrn_model(model_path)
    
    if isinstance(sp, str):
        sp_proc = load_spm_model(sp)
    else:
        sp_proc = sp 

    input_tokens = sp_proc.encode(prompt)
    encoder_input = pad_sequences([input_tokens], maxlen=max_len, padding="post")
    seq = [START_TOKEN_ID] + input_tokens

    for _ in range(max_generate):
        decoder_input = np.array([seq])
        preds = model.predict([encoder_input, decoder_input], verbose=0)
        probs = preds[0, -1, :] 

    # Get top 3 token IDs (highest probabilities first)
    top_3_indices = np.argsort(probs)[-3:][::-1]
    
    # Decode each token individually to get separate word predictions
    predictions = []
    for token_id in top_3_indices:
        word = sp_proc.decode([int(token_id)])
        if word.strip():  # Filter out empty tokens
            predictions.append(word.strip())
    
    text = " ".join(predictions[:max_generate])
    
    return text


# ==================== Bidirectional Prediction Function ====================
def get_top_3_preds_bidirectional(prompt, sp="autocomplete/tausug_spm.model", model_path="autocomplete/BIDIRECTIONAL-FINETUNED2.keras", max_len=MAX_LEN, max_generate=3):
    """
    Get top-3 next word predictions using Bidirectional model
    
    Args:
        prompt: Input text
        sp: SentencePiece model path
        model_path: Bidirectional model checkpoint path
        max_len: Maximum sequence length
        max_generate: Number of predictions
    """
    model = load_rrn_model(model_path)
    
    if isinstance(sp, str):
        sp_proc = load_spm_model(sp)
    else:
        sp_proc = sp 

    input_tokens = sp_proc.encode(prompt)
    encoder_input = pad_sequences([input_tokens], maxlen=max_len, padding="post")
    seq = [START_TOKEN_ID] + input_tokens

    for _ in range(max_generate):
        decoder_input = np.array([seq])
        preds = model.predict([encoder_input, decoder_input], verbose=0)
        probs = preds[0, -1, :] 

    # Get top 3 token IDs (highest probabilities first)
    top_3_indices = np.argsort(probs)[-3:][::-1]
    
    # Decode each token individually to get separate word predictions
    predictions = []
    for token_id in top_3_indices:
        word = sp_proc.decode([int(token_id)])
        if word.strip():  # Filter out empty tokens
            predictions.append(word.strip())
    
    text = " ".join(predictions[:max_generate])
    
    return text


# ==================== GRU Prediction Function ====================
def get_top_3_preds_gru(prompt, sp="autocomplete/gru_spm.model", model_path="autocomplete/GRU.pt", max_generate=3):
    """
    Get top-3 next word predictions using PyTorch GRU model
    
    Args:
        prompt: Input text
        sp: SentencePiece model path (uses gru_spm.model with 5000 vocab by default)
        model_path: GRU model checkpoint path
        max_generate: Number of predictions
    """
    # Load model
    model, config = load_gru_model(model_path)
    
    if isinstance(sp, str):
        sp_proc = load_spm_model(sp)
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
        probs = torch.softmax(last_logits, dim=-1)
    
    # Get top 3 token IDs
    top_3_probs, top_3_indices = torch.topk(probs, k=min(max_generate, len(probs)))
    
    # Decode predictions
    predictions = []
    for token_id in top_3_indices.tolist():
        # Skip special tokens
        if token_id in [PAD_TOKEN, START_TOKEN_ID, END_TOKEN_ID]:
            continue
        
        word = sp_proc.decode([int(token_id)])
        if word.strip():  # Filter out empty tokens
            predictions.append(word.strip())
    
    # Join top predictions
    text = " ".join(predictions[:max_generate])
    
    return text
