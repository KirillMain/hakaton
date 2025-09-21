from rest_framework import serializers
from help.models import ArticleNew


class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = ArticleNew
        fields = (
            "title",
            "url",
        )
