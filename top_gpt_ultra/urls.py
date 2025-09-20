from django.urls import path, include
from top_gpt_ultra.views import Alive

urlpatterns = [
    path("alive/", Alive.as_view()),
    path("api/v1/profile/", include("user_profile.urls")),
    path("api/v1/assistant/",include("assistant.urls")),
]
