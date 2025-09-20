from django.urls import path, include
from assistant.views import ParseQueryView


urlpatterns = [
    path("parse/", ParseQueryView.as_view()),
]
