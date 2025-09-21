from django.db import models
import uuid
from django.core.validators import MinValueValidator, MaxValueValidator


class QueryLog(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    # user_id=models.UUIDField(primary_key=True, default=uuid.uuid4)
    query_text = models.TextField()
    normalized_text = models.TextField()
    intent = models.CharField(
        max_length=100,
        verbose_name="Интент",
        help_text="Распознанное намерение пользователя",
    )
    # ACTION_TYPES = [("search", "Поиск"), ("action", "Действие")]
    action_type = models.CharField(
        max_length=32, verbose_name="Тип действия", blank=True, null=True
    )
    entities = models.JSONField(
        verbose_name="Сущности",
        default=dict,
        blank=True,
    )
    confidence = models.FloatField(
        verbose_name="Уверенность",
        default=0.0,
        help_text="Уверенность модели в распознавании",
        validators=[
            MinValueValidator(0.0, "Уверенность не может быть отрицательной"),
            MaxValueValidator(1.0, "Уверенность не может превышать 1.0"),
        ],
    )
    created_at = models.DateTimeField(
        auto_now_add=True, verbose_name="Дата создания"
    )
    rate = models.PositiveIntegerField(default=None, null=True)
