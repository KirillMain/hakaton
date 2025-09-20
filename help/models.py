from django.db import models
import uuid
from django.contrib.postgres.fields import ArrayField


class Article(models.Model):
    id = models.UUIDField(default=uuid.uuid4, primary_key=True)
    title = models.CharField(max_length=255, verbose_name="Заголовок")
    url = models.URLField(verbose_name="URL статьи", unique=True)



