from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# from assistant.classifier_mock_utils import classify_intent
from assistant.models import QueryLog
from assistant.nlp.intent_service import analyze_query
from assistant.serializer import QueryLogSerializer


class ParseQueryView(APIView):
    def post(self, request):
        query_text = request.data.get("query", "")
        # result = classify_intent(query_text)
        result = analyze_query(query_text)

        QueryLog.objects.create(
            query_text=query_text,
            # normalized_text=result.get("normalized_text"),
            normalized_text=result.get("input"),
            intent=result.get("intent"),
            action_type=result.get("action_type"),
            confidence=result.get("intent_confidence"),
            entities=result.get("entities"),
        )

        # return Response(data=QueryLogSerializer(q).data, status=status.HTTP_200_OK)
        return Response(data=result, status=status.HTTP_200_OK)
