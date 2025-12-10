import sentencepiece as spm
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import keras
from pathlib import Path
import tensorflow as tf
from keras.saving import register_keras_serializable

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
