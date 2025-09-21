from django.urls import path, include
from assistant.views import ParseQueryView


urlpatterns = [
    path("query/", ParseQueryView.as_view()),
    path("query/<uuid:query_id>/rate/<int:rate>/", ParseQueryView.as_view()),
]
