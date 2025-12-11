from django.apps import AppConfig


class AutocompleteConfig(AppConfig):
    name = 'autocomplete'
    
    def ready(self):
        """
        Called once at Django startup.
        Preloads vocabulary and processors.
        Models load on-demand to save memory.
        """
        # Import here to avoid AppRegistryNotReady error
        from .utils import model_cache
        
        # Preload all models on server startup
        try:
            model_cache.preload_all_models()
        except Exception as e:
            print(f"[WARNING] Failed to preload models on startup: {e}")
            print("Models will be loaded on first request instead.")
        
        print("\n" + "=" * 60)
        print("VOCABULARY AND PROCESSORS LOADED!")
        print("Preloading lightweight model to avoid timeouts...")
        print("=" * 60 + "\n")
        
        # Preload the smallest model to avoid first-request timeout
        try:
            from .ml_models import load_lstm_model
            load_lstm_model()  # This is typically the smallest/fastest
            print("✓ LSTM model preloaded successfully\n")
        except Exception as e:
            print(f"⚠ Model preload warning: {e}\n")

