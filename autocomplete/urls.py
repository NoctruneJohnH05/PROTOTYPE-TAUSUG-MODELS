from django.urls import path
from . import views

urlpatterns = [
    path("", views.render_doc_editor, name="home"),  

    path("document/", views.render_doc_editor, name="render-doc-editor"),

    path("generate/", views.generate_text, name="generate-text"),

    # autocomplete endpoint (must exist in views.py)
    path("autocomplete/", views.autocomplete, name="autocomplete"),

    # translate endpoint (must exist in views.py)
    path("translate/", views.translate, name="translate"),

    # prediction endpoint
    path("predict/", views.predict_next_word, name="predict_next_word"),
]
