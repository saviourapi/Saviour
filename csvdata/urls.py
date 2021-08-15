from django.urls import path
from .views import *

urlpatterns = [
    path('upload/', load_data),
    path('columns/', get_csv_header),
]
