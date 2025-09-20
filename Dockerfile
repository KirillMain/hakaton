FROM python:3.13-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["cd", "top_gpt_ultra"]
CMD ["gunicorn", "top_gpt_ultra.wsgi:application", "--bind", "0.0.0.0:8000"]