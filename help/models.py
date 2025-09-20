from django.db import models
import uuid
from django.contrib.postgres.fields import ArrayField


class Keyword(models.Model):
    id = models.UUIDField(default=uuid.uuid4, primary_key=True)
    key = models.CharField(
        max_length=100, unique=True, verbose_name="Ключевое слово"
    )


class Article(models.Model):
    id = models.UUIDField(default=uuid.uuid4, primary_key=True)
    title = models.CharField(max_length=255, verbose_name="Заголовок")
    url = models.URLField(verbose_name="URL статьи", unique=True)
    tags = ArrayField(
        models.CharField(max_length=50), blank=True, default=list
    )
    embedding = ArrayField(
        models.FloatField(),
        size=384,
        blank=True,
        null=True,
        verbose_name="Векторное представление",
    )


class ArticleKeyword(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    article = models.ForeignKey(
        Article,
        on_delete=models.CASCADE,
        related_name="article_keywords",
        verbose_name="Статья",
    )
    keyword = models.ForeignKey(
        Keyword,
        on_delete=models.CASCADE,
        related_name="article_keywords",
        verbose_name="Ключевое слово",
    )
    weight = models.FloatField(
        verbose_name="Вес ключевого слова",
    )
