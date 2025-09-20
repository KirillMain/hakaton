from rest_framework.views import APIView
from user_profile.models import UserProfile
from rest_framework.response import Response
from .serializers import UserProfileSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework import status


class ProfileView(APIView):

    def get(
        self,
        request,
    ):
        profile = UserProfile.objects.get(user=request.user)
        serializer = UserProfileSerializer(profile)
        return Response(serializer.data)

    def patch(self, request):

        profile = UserProfile.objects.get(user=request.user)
        serializer = UserProfileSerializer(
            profile, data=request.data, partial=True
        )

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
