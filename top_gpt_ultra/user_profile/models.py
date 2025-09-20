from django.db import models
import uuid
from django.contrib.auth.models import User


class UserProfile(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        primary_key=True,
        verbose_name="Пользователь",
        related_name="profile",
    )
    inn = models.CharField(max_length=12, blank=True, null=True)
    slice_date = models.DateField(
        null=True,
    )
    settings = models.JSONField(
        verbose_name="Настройки пользователя", default=dict, blank=True
    )
