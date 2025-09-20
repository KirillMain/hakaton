from django.db import models
import uuid


class UserProfile(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    inn = models.CharField(max_length=12, blank=True, null=True)
    slice_date = models.DateField(
        null=True,
    )
    settings = models.JSONField(
        verbose_name="Настройки пользователя", default=dict, blank=True
    )
