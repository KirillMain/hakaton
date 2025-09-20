from django.urls import path, include

urlpatterns = [
    path("api/v1/profile/", include("user_profile.urls")),
    path("api/v1/assistant/",include("assistant.urls")),
]
