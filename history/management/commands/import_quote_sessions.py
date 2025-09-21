import pandas as pd
from django.core.management.base import BaseCommand
from history.models import QuoteSessionHistory


class Command(BaseCommand):
    help = "Import procurement data from Excel file"

    def handle(self, *args, **options):
        file_path = (
            "D:\programming\hakaton\\config\history\management\commands\qs.xlsx"
        )

        df = pd.read_excel(file_path, sheet_name="Лист1")
        df = df.rename(
            columns={
                "Наименование КС": "quote_name",
                "ID КС": "quote_id",
                "Сумма КС": "quote_price",
                "Дата создания КС": "quote_created_at",
                "Дата завершения КС": "quote_ended_at",
                "Категория ПП первой позиции спецификации": "category_name",
                "Наименование заказчика": "customer_name",
                "ИНН заказчика": "customer_inn",
                "Наименование поставщика": "vendor_name",
                "ИНН поставщика": "vendor_inn",
                "Закон-основание": "fundamental_law",
            }
        )

        df["quote_created_at"] = pd.to_datetime(df["quote_created_at"]).dt.date
        df["quote_ended_at"] = pd.to_datetime(df["quote_ended_at"]).dt.date

        df = df.fillna("")

        error_ids = []
        for _, row in df.iterrows():
            try:
                if QuoteSessionHistory.objects.filter(
                    quote_id=row["quote_id"]
                ).exists():
                    self.stdout.write(
                        self.style.SUCCESS(f'Skipped ID {row["quote_id"]}')
                    )
                    continue
                QuoteSessionHistory.objects.create(
                    quote_name=row["quote_name"],
                    quote_id=row["quote_id"],
                    quote_price=row["quote_price"],
                    quote_created_at=row["quote_created_at"],
                    quote_ended_at=row["quote_ended_at"],
                    category_name=row["category_name"],
                    customer_name=row["customer_name"],
                    customer_inn=row["customer_inn"],
                    vendor_name=row["vendor_name"],
                    vendor_inn=row["vendor_inn"],
                    fundamental_law=row["fundamental_law"],
                )

                self.stdout.write(
                    self.style.SUCCESS(
                        f'Successfully imported {row["quote_name"]} with ID {row["quote_id"]}'
                    )
                )
            except:
                error_ids.append(row["quote_id"])
                self.stdout.write(
                    self.style.ERROR(
                        f'Unsuccessfully imported {row["quote_name"]} with ID {row["quote_id"]}'
                    )
                )

        self.stdout.write(
            self.style.WARNING(f"Error_ids: {error_ids}")
        )  # Error_ids: [10041638]
