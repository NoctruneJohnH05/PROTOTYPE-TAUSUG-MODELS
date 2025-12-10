from django.urls import path
from . import views


urlpatterns = [
    path("document/", views.render_doc_editor, name="render-doc-editor"),
]