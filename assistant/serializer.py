from rest_framework import serializers
from assistant.models import QueryLog


class QueryLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = QueryLog
        fields = "__all__"
