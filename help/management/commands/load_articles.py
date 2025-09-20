import json
from django.core.management.base import BaseCommand
from help.models import Article
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
                    urls = self.find_all_urls_in_article(article)  
                    print(urls)
                    for url in urls:
                        simplified_article = {
                            "largeName": article.get('largeName', ''),
                            "url": url,
                        }
                        simplified_data.append(simplified_article)
        print(simplified_data)

    def find_all_urls_in_article(self, article):
        """Ищет ВСЕ URL во всех строковых полях статьи"""
        all_urls = []
        patterns = [
            r'https://zakupki\.mos\.ru/knowledgebase/article/details/cms/[a-zA-Z0-9]+',
            r'https://zakupki.mos.ru/knowledgebase/article/details/cms/[a-zA-Z0-9]+',
        ]
        
        for key, value in article.items():
            if isinstance(value, str):
                for pattern in patterns:
                    matches = re.findall(pattern, value)  # findall вместо search
                    all_urls.extend(matches)
        
        # Удаляем дубликаты
        return list(set(all_urls))
        
        
        
        
"""     
for item_data in simplified_data:
            try:
                Article.objects.get_or_create(
                    url=item_data.get("url"),  
                    defaults={
                        'title': item_data.get("largeName")  
                    }
                )
            except :
                pass
"""
