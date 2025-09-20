import pandas as pd
from django.core.management.base import BaseCommand
from history.models import ContractHistory


class Command(BaseCommand):
    help = "Import procurement data from Excel file"

    def handle(self, *args, **options):
        file_path = (
            "D:\programming\hakaton\\top_gpt_ultra\history\management\commands\c.xlsx"
        )

        df = pd.read_excel(file_path, sheet_name="Лист1")
        df = df.rename(
            columns={
                "Наименование контракта": "contract_name",
                "ID контракта": "contract_id",
                "Сумма контракта": "contract_price",
                "Дата заключения контракта": "created_at",
                "Категория ПП первой позиции спецификации": "category_name",
                "Наименование заказчика": "customer_name",
                "ИНН заказчика": "customer_inn",
                "Наименование поставщика": "vendor_name",
                "ИНН поставщика": "vendor_inn",
                "Закон-основание": "fundamental_law",
            }
        )

        df["created_at"] = pd.to_datetime(df["created_at"]).dt.date

        df = df.fillna("")

        error_ids = []
        for _, row in df.iterrows():
            try:
                if ContractHistory.objects.filter(
                    contract_id=row["contract_id"]
                ).exists():
                    self.stdout.write(
                        self.style.SUCCESS(f'Skipped ID {row["contract_id"]}')
                    )
                    continue
                ContractHistory.objects.create(
                    contract_name=row["contract_name"],
                    contract_id=row["contract_id"],
                    contract_price=row["contract_price"],
                    created_at=row["created_at"],
                    category_name=row["category_name"],
                    customer_name=row["customer_name"],
                    customer_inn=row["customer_inn"],
                    vendor_name=row["vendor_name"],
                    vendor_inn=row["vendor_inn"],
                    fundamental_law=row["fundamental_law"],
                )

                self.stdout.write(
                    self.style.SUCCESS(
                        f'Successfully imported {row["contract_name"]} with ID {row["contract_id"]}'
                    )
                )
            except Exception as e:
                error_ids.append(row["contract_id"])
                self.stdout.write(
                    self.style.ERROR(
                        f'Unsuccessfully imported {row["contract_name"]} with ID {row["contract_id"]} {str(e)}'
                    )
                )

        self.stdout.write(
            self.style.WARNING(f"Error_ids: {error_ids}")
        )  # Error_ids: []
