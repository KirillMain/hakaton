from django.db import models
import uuid
from django.contrib.auth.models import User


class UserProfile(models.Model):
    slice_date = models.DateField(
        null=True,
    )
    settings = models.JSONField(
        verbose_name="Настройки пользователя", default=dict, blank=True
    )
