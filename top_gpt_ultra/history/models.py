from django.db import models


class QuoteSessionHistory(models.Model):
    quote_name = models.CharField(max_length=2048)
    quote_id = models.PositiveBigIntegerField()
    quote_price = models.FloatField()
    quote_created_at = models.DateField()
    quote_ended_at = models.DateField()
    category_name = models.CharField(max_length=1024)
    customer_name = models.CharField(max_length=1024)
    customer_inn = models.PositiveBigIntegerField()
    vendor_name = models.CharField(max_length=1024)
    vendor_inn = models.PositiveBigIntegerField()
    fundamental_law = models.CharField(max_length=64)


class ContractHistory(models.Model):
    contract_name = models.CharField(max_length=2048)
    contract_id = models.PositiveBigIntegerField()
    contract_price = models.FloatField()
    created_at = models.DateField()
    category_name = models.CharField(max_length=1024)
    customer_name = models.CharField(max_length=1024)
    customer_inn = models.PositiveBigIntegerField()
    vendor_name = models.CharField(max_length=1024)
    vendor_inn = models.PositiveBigIntegerField()
    fundamental_law = models.CharField(max_length=64)
