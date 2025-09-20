from django.urls import path, include
from user_profile.views import ProfileView


urlpatterns = [
    path("",ProfileView.as_view())
]
