from pathlib import Path
import requests
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime
import re


BASE_DIR = Path(__file__).resolve().parent


class ArticleVectorizer:
    def __init__(self, model_path=None):
        """
        Инициализация векторного поиска статей
        model_path: путь к локальной модели SentenceTransformer
        """
        # Загружаем модель (будет использовать локальные файлы)
        if model_path:
            self.model = SentenceTransformer(model_path)
        else:
            # Используем стандартную многоязычную модель
            self.model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

        self.articles = []
        self.embeddings = None
        self.index = None

    def fetch_articles(self, url):
        """
        Парсинг статей с указанного URL
        """
        print(f"Загружаем статьи с {url}...")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            for cat in data:
                for articles_by_service in cat["articlesByService"]:
                    for article in articles_by_service["articles"]:
                        # Обрабатываем разные форматы статей
                        article_id = article.get("id")
                        title = article.get("largeName")
                        content = (
                            article.get("instructionOperator")
                            if article.get("detailText") == ""
                            else article.get("detailText")
                        )

                        # Очищаем HTML теги если они есть
                        print("qqweqweq", content)
                        title = self.clean_html(title)
                        content = self.clean_html(content)
                        print("zxcxzczxc", content)

                        # Создаем текст для эмбеддинга (заголовок + содержание)
                        text_for_embedding = f"{title}. {content}"  # Ограничиваем длину

                        self.articles.append(
                            {
                                "id": article_id,
                                "title": title,
                                "content": content,
                                "text_for_embedding": text_for_embedding,
                            }
                        )

            print(f"Загружено {len(self.articles)} статей")
            return True

        except Exception as e:
            print(f"Ошибка при загрузке статей: {e}")
            return False

    def clean_html(self, text):
        """Очистка HTML тегов из текста"""
        if not text:
            return ""
        import html

        decoded_content = html.unescape(text)
        clean_text = re.sub(r"<[^>]+>", " ", decoded_content)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        return clean_text

    def create_embeddings(self):
        """
        Создание векторных представлений для всех статей
        """
        if not self.articles:
            print("Нет статей для обработки")
            return False

        print("Создаем векторные представления...")

        # Подготавливаем тексты для кодирования
        texts = [article["text_for_embedding"] for article in self.articles]

        # Создаем эмбеддинги
        self.embeddings = self.model.encode(
            texts, convert_to_tensor=False, show_progress_bar=True, batch_size=32
        )

        # Преобразуем в numpy array для FAISS
        self.embeddings = np.array(self.embeddings).astype("float32")

        print(f"Создано {len(self.embeddings)} векторных представлений")
        return True

    def build_index(self):
        """
        Построение FAISS индекса
        """
        if self.embeddings is None:
            print("Сначала создайте эмбеддинги")
            return False

        print("Строим FAISS индекс...")

        # Размерность вектора
        dimension = self.embeddings.shape[1]

        # Создаем индекс (используем L2 расстояние)
        self.index = faiss.IndexFlatL2(dimension)

        # Добавляем векторы в индекс
        self.index.add(self.embeddings)

        print(
            f"Индекс построен. Размерность: {dimension}, количество векторов: {self.index.ntotal}"
        )
        return True

    def save_data(self, base_path="article_data"):
        """
        Сохранение всех данных
        """
        import os

        os.makedirs(base_path, exist_ok=True)

        # Сохраняем статьи
        with open(f"{base_path}/articles.json", "w", encoding="utf-8") as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)

        # Сохраняем FAISS индекс
        if self.index is not None:
            faiss.write_index(self.index, f"{base_path}/articles_faiss.index")

        # Сохраняем метаданные
        metadata = {
            "created_at": datetime.now().isoformat(),
            "articles_count": len(self.articles),
            "embedding_dimension": (
                self.embeddings.shape[1] if self.embeddings is not None else None
            ),
            "model_name": str(self.model),
        }

        with open(f"{base_path}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"Данные сохранены в папку {base_path}")

    def load_data(self, base_path="article_data"):
        """
        Загрузка ранее сохраненных данных
        """
        try:
            # Загружаем статьи
            with open(f"{base_path}/articles.json", "r", encoding="utf-8") as f:
                self.articles = json.load(f)

            # Загружаем FAISS индекс
            self.index = faiss.read_index(f"{base_path}/articles_faiss.index")

            print(
                f"Загружено {len(self.articles)} статей и индекс с {self.index.ntotal} векторами"
            )
            return True

        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return False

    def search_articles(self, query, top_k=5):
        """
        Поиск статей по запросу
        """
        if self.index is None or not self.articles:
            print("Индекс не загружен")
            return []

        # Преобразуем запрос в вектор
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype("float32")

        # Ищем ближайшие векторы
        distances, indices = self.index.search(query_embedding, top_k)

        # Формируем результаты
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.articles):
                article = self.articles[idx]
                results.append(
                    {
                        "article": article,
                        "distance": float(distances[0][i]),
                        "score": 1
                        / (
                            1 + distances[0][i]
                        ),  # Преобразуем расстояние в оценку (0-1)
                    }
                )

        return results


# Основной скрипт выполнения
def main():
    # URL для парсинга
    articles_url = "https://zakupki.mos.ru/newapi/api/KnowledgeBase/GetArticlesBySectionType?sectionType=supplier"

    # Инициализируем векторizer
    vectorizer = ArticleVectorizer()

    # Парсим статьи
    if vectorizer.fetch_articles(articles_url):
        # Создаем эмбеддинги
        if vectorizer.create_embeddings():
            # Строим индекс
            if vectorizer.build_index():
                # Сохраняем всё
                vectorizer.save_data(str(BASE_DIR) + "\\mos_zakupki_articles")

                # Тестовый поиск
                test_query = "как подать заявку на участие"
                results = vectorizer.search_articles(test_query, top_k=3)

                print(f"\nРезультаты поиска для '{test_query}':")
                for i, result in enumerate(results, 1):
                    print(
                        f"{i}. {result['article']['title']} (сходство: {result['score']:.3f})"
                    )


def search_example():
    # Инициализируем загрузчик
    vectorizer = ArticleVectorizer()

    # Загружаем сохраненные данные
    if vectorizer.load_data(str(BASE_DIR) + "\\mos_zakupki_articles"):
        while True:
            query = input("\nВведите поисковый запрос (или 'quit' для выхода): ")
            if query.lower() == "quit":
                break

            results = vectorizer.search_articles(query, top_k=5)

            print(f"\nНайдено статей: {len(results)}")
            for i, result in enumerate(results, 0):
                article = result["article"]
                print(f"{i+1}. [{result['score']:.6f}] {article['title']}")
                print(f"   ID: {article['id']}")


# if __name__ == "__main__":
#     main()


if __name__ == "__main__":
    search_example()
