from django.apps import AppConfig
import os


class AutocompleteConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'autocomplete'
    
    def ready(self):
        """
        Preloads vocabulary and processors only.
        Models load on-demand to save memory.
        """
        if os.environ.get('RUN_MAIN') != 'true':
            return
        
        from .ml_models import preload_vocab_and_processors
        
        print("\n" + "=" * 60)
        print("PRELOADING ESSENTIAL RESOURCES...")
        print("=" * 60)
        
        preload_vocab_and_processors()
        
        print("=" * 60)
        print("VOCABULARY AND PROCESSORS LOADED!")
        print("Models will load on-demand to save memory.")
        print("=" * 60 + "\n")

