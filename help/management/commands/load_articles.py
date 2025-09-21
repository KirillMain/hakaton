import json
from django.core.management.base import BaseCommand
from help.models import ArticleNew
from django.core.exceptions import MultipleObjectsReturned
import re

class Command(BaseCommand):

    def handle(self, *args, **options):
        path="/home/danil/haha/hakaton/GetArticlesBySectionType.json"
        with open(path,'r') as file:
            data=json.load(file)

        simplified_data = []
    
        
        for category in data:
            for service in category.get('articlesByService', []):
                for article in service.get('articles', []):
                    simplified_article = {
                    "id": article.get("id",""),
                    "largeName": article.get('largeName', ''),
                    "url": f"https://zakupki.mos.ru/knowledgebase/article/details/ais/{article.get('id', '')}"
                    }
                    simplified_data.append(simplified_article)

        for item_data in simplified_data:
            try:
                ArticleNew.objects.get_or_create(
                    url=item_data.get("url"),  
                    defaults={
                        'title': item_data.get("largeName"),
                        'id': item_data.get("id")
                    }
                )
            except :
                pass

