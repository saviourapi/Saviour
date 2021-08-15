from django.urls import path
from .views import *

urlpatterns = [
    path('', learn),
    path('predict/', predict),
    path('delete/', delete_model),
    # path('meta/<str:id>/', get_model_meta),
]
