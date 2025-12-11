from django.urls import path
from . import views


urlpatterns = [
    path("", views.render_doc_editor, name="home"),  # Root URL
    path("document/", views.render_doc_editor, name="render-doc-editor"),
    path("generate/", views.generate_text, name="generate-text"),
    path('autocomplete/', views.autocomplete, name='autocomplete'),
    path('translate/', views.translate, name='translate'),
    path('document/', views.predict_next_word, name='predict_next_word'),  # For prediction
    # Add any other API endpoints you need
]
