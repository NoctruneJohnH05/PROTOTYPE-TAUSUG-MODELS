from django.urls import path
from . import views

urlpatterns = [
    # Root page
    path("", views.render_doc_editor, name="home"),

    # Editor + prediction models
    path("document/", views.render_doc_editor, name="render-doc-editor"),
    path("predict/", views.predict_next_word, name="predict_next_word"),

    # Text generation
    path("generate/", views.generate_text, name="generate-text"),

    # Placeholder endpoints (must exist)
    path("autocomplete/", views.autocomplete, name="autocomplete"),
    path("translate/", views.translate, name="translate"),
]
