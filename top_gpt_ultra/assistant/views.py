from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from assistant.classifier_mock_utils import classify_intent
from assistant.models import QueryLog
from assistant.serializer import QueryLogSerializer


class ParseQueryView(APIView):
    def post(self, request):
        query_text = request.data.get("query", "")
        context = request.data.get("context", {})

        result = classify_intent(query_text)

        intent = result.get("intent")
        action_type = result.get("action_type")
        confidence = float(result.get("confidence", 0.0))

        q = QueryLog.objects.create(
            query_text=query_text,
            normalized_text=result.get("normalized_text"), 
            intent=result.get("intent"), 
            action_type=result.get("action_type"), 
            confidence=result.get("confidence"),
            entities=result.get("matches"),
        )

        return Response(data=QueryLogSerializer(q).data, status=status.HTTP_200_OK)
