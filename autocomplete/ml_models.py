import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import gc
import tensorflow as tf

# Enable TensorFlow memory optimization
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_soft_device_placement(True)

# Track last used model to enable cleanup
_last_used_model = None

def cleanup_unused_models(keep_model=None):
    """Unload models from memory except the one being used"""
    global _loaded_models, _last_used_model
    
    for model_name in list(_loaded_models.keys()):
        if model_name != keep_model and _loaded_models[model_name] is not None:
            print(f"[MEMORY] Unloading {model_name} model to free RAM")
            _loaded_models[model_name] = None
            gc.collect()
            if 'tensorflow' in str(type(_loaded_models.get(model_name))):
                tf.keras.backend.clear_session()
    
    _last_used_model = keep_model

def load_lstm_model():
    global _loaded_models, _last_used_model
    
    if _loaded_models['lstm'] is None:
        print("[LSTM] Loading model...")
        cleanup_unused_models(keep_model='lstm')
        
        try:
            _loaded_models['lstm'] = tf.keras.models.load_model(
                'autocomplete/tausug_next_word_lstm.h5',
                compile=False
            )
            print("[LSTM] ✓ Model loaded successfully")
            _last_used_model = 'lstm'
        except Exception as e:
            print(f"[LSTM] ✗ Error: {e}")
            raise
    
    return _loaded_models['lstm']

def load_bidirectional_model():
    global _loaded_models, _last_used_model
    
    if _loaded_models['bidirectional'] is None:
        print("[BIDIRECTIONAL] Loading model...")
        cleanup_unused_models(keep_model='bidirectional')
        
        try:
            _loaded_models['bidirectional'] = tf.keras.models.load_model(
                'autocomplete/tausug_bidirectional_model.h5',
                compile=False
            )
            print("[BIDIRECTIONAL] ✓ Model loaded successfully")
            _last_used_model = 'bidirectional'
        except Exception as e:
            print(f"[BIDIRECTIONAL] ✗ Error: {e}")
            raise
    
    return _loaded_models['bidirectional']

def load_gru_model():
    global _loaded_models, _last_used_model
    
    if _loaded_models['gru'] is None:
        print("[GRU] Loading model...")
        cleanup_unused_models(keep_model='gru')
        
        try:
            _loaded_models['gru'] = torch.load(
                'autocomplete/tausug_gru_model.pth',
                map_location=torch.device('cpu')
            )
            _loaded_models['gru'].eval()
            print("[GRU] ✓ Model loaded successfully")
            _last_used_model = 'gru'
        except Exception as e:
            print(f"[GRU] ✗ Error: {e}")
            raise
    
    return _loaded_models['gru']