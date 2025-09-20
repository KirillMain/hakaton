from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from assistant.classifier_mock_utils import classify_intent


class ParseQueryView(APIView):
    def post(self, request):
        query = request.data.get("query", "").strip()
        context = request.data.get("query", {})
        result = classify_intent(query)
        intent = result.get("intent")
        action_type = result.get("action_type")
        confidence = float(result.get("confidence", 0.0))
