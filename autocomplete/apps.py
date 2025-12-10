from django.apps import AppConfig


class AutocompleteConfig(AppConfig):
    name = 'autocomplete'
    
    def ready(self):
        """
        This method is called when Django starts.
        Preload all models into memory for faster predictions.
        """
        # Import here to avoid AppRegistryNotReady error
        from .utils import model_cache
        
        # Preload all models on server startup
        try:
            model_cache.preload_all_models()
        except Exception as e:
            print(f"[WARNING] Failed to preload models on startup: {e}")
            print("Models will be loaded on first request instead.")

