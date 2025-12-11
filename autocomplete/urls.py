from django.urls import path
from . import views


urlpatterns = [
    path("", views.render_doc_editor, name="home"),  # Root URL
    path("document/", views.render_doc_editor, name="render-doc-editor"),
    path("generate/", views.generate_text, name="generate-text"),
]
