from rest_framework import serializers
from history.models import QuoteSessionHistory, ContractHistory


class QuoteSessionHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = QuoteSessionHistory
        fields = "__all__"


class ContractHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = ContractHistory
        fields = "__all__"
