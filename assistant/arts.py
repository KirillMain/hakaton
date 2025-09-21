from pathlib import Path
import requests
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime
import re
import os


BASE_DIR = Path(__file__).resolve().parent

# Глобальная переменная для хранения инициализированной модели
_article_vectorizer = None

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

            unique_ids = []

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

                        if article_id in unique_ids:
                            print(f"Найден дубликат статьи с ID {article_id}, пропускаем")
                            continue
                        else:
                            unique_ids.append(article_id)

                        # Очищаем HTML теги если они есть
                        title = self.clean_html(title)
                        content = self.clean_html(content)

                        # Создаем текст для эмбеддинга (заголовок + содержание)
                        text_for_embedding = f"{title}. {content}"

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

    def search_articles_ids(self, query, top_k=5):
        """
        Поиск статей по запросу и возврат только ID статей
        """
        if self.index is None or not self.articles:
            print("Индекс не загружен")
            return []

        # Преобразуем запрос в вектор
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype("float32")

        # Ищем ближайшие векторы
        distances, indices = self.index.search(query_embedding, top_k)

        # Формируем результаты - только ID статей
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.articles):
                article = self.articles[idx]
                results.append({
                    "id": article["id"],
                    "score": 1 / (1 + distances[0][i]),  # Преобразуем расстояние в оценку (0-1)
                    "distance": float(distances[0][i])
                })

        return results

    def search_articles(self, query, top_k=5):
        """
        Поиск статей по запросу (полная информация)
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
                        "score": 1 / (1 + distances[0][i]),
                    }
                )

        return results


def init_model():
    """
    Инициализация модели при запуске сервера Django
    Проверяет, не была ли модель уже инициализирована
    """
    global _article_vectorizer
    
    if _article_vectorizer is not None:
        print("Модель уже инициализирована")
        return _article_vectorizer
    
    print("Инициализация модели...")
    
    # Создаем экземпляр векторного поиска
    vectorizer = ArticleVectorizer()
    
    # Путь к сохраненным данным
    data_path = os.path.join(BASE_DIR, "mos_zakupki_articles")
    
    # Проверяем существование сохраненных данных
    if os.path.exists(data_path):
        print("Загрузка сохраненных данных...")
        if vectorizer.load_data(data_path):
            _article_vectorizer = vectorizer
            print("Модель успешно инициализирована с загруженными данными")
            return vectorizer
        else:
            print("Не удалось загрузить сохраненные данные")
    else:
        print("Сохраненные данные не найдены")
        print("Загрузка новых данных")
        articles_url = "https://zakupki.mos.ru/newapi/api/KnowledgeBase/GetArticlesBySectionType?sectionType=supplier"
        if vectorizer.fetch_articles(articles_url) and vectorizer.create_embeddings() and vectorizer.build_index():
            vectorizer.save_data(data_path)
            _article_vectorizer = vectorizer
            return vectorizer
    
    _article_vectorizer = vectorizer
    return vectorizer


def search_articles_by_text(text, top_k=5):
    """
    Функция для поиска статей по тексту, возвращает только ID статей
    """
    global _article_vectorizer
    
    # Если модель не инициализирована, инициализируем её
    if _article_vectorizer is None:
        _article_vectorizer = init_model()
    
    # Выполняем поиск и возвращаем только ID
    results = _article_vectorizer.search_articles_ids(text, top_k)
    
    # Возвращаем список ID
    return [result["id"] for result in results]


def search_articles_with_scores(text, top_k=5):
    """
    Функция для поиска статей по тексту с возвратом ID и оценок
    """
    global _article_vectorizer
    
    # Если модель не инициализирована, инициализируем её
    if _article_vectorizer is None:
        _article_vectorizer = init_model()
    
    # Выполняем поиск и возвращаем ID с оценками
    return _article_vectorizer.search_articles_ids(text, top_k)

def get_serialized_arts_by_text(text):
    from help.models import ArticleNew
    from help.serializer import ArticleSerializer

    art_ids = search_articles_by_text(text)


    arts_data = []
    for art_id in art_ids:
        art = ArticleNew.objects.filter(id=art_id)
        if art.exists():
            arts_data.append(ArticleSerializer(art.first()).data)
        else:
            print(f"Статьи с ID {art_id} нет в бд!!!")

    return arts_data


