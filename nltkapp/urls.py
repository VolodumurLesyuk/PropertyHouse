from django.urls import path
from .viewsets import TokenizeTextView, POSTagView, NERView

urlpatterns = [
    path('tokenize/', TokenizeTextView.as_view(), name='tokenize-text'),
    path('pos_tag/', POSTagView.as_view(), name='pos_tag'),
    path('ner/', NERView.as_view(), name='ner'),
]
