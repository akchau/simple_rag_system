import requests
import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# 1. Модель E5-small: идеальный баланс скорости и точности для RTX 4060
model = SentenceTransformer("intfloat/multilingual-e5-small")

def get_context(query, k=5): 
    """Поиск релевантных кусков текста в векторной базе."""
    if not os.path.exists("index.faiss"): return "Контекст не найден"
    index = faiss.read_index("index.faiss")
    with open("docs.pkl", "rb") as f: docs = pickle.load(f)
    
    # Модель E5 требует префикса 'query: ' для поисковых запросов
    v = model.encode(["query: " + query])
    dist, idx = index.search(np.array(v).astype("float32"), k)
    return "\n---\n".join([docs[i] for i in idx[0] if i < len(docs)])

def ask_ollama(q, ctx):
    """Отправка запроса в Llama 3 с жесткими инструкциями против галлюцинаций."""
    prompt = (
        f"### РОЛЬ: Ты — робот-архивариус компании Aethelgard. "
        f"### ПРАВИЛО: Отвечай ТОЛЬКО используя информацию из предоставленного КОНТЕКСТА. "
        f"Если в КОНТЕКСТЕ нет прямого упоминания факта — отвечай: 'В предоставленной документации информация отсутствует'. "
        f"ЗАПРЕЩЕНО использовать внешние знания о кибербезопасности или законах.\n\n"
        f"### КОНТЕКСТ:\n{ctx}\n\n"
        f"### ВОПРОС: {q}\n\n"
        f"### ОТВЕТ НА РУССКОМ:"
    )
    
    try:
        r = requests.post("http://localhost:11434/api/generate", 
            json={
                "model": "llama3", 
                "prompt": prompt, 
                "stream": False,
                "options": {
                    "num_ctx": 16384,     # Увеличенное окно контекста для 50 МБ базы
                    "temperature": 0.0,    # Максимальная точность, запрет на фантазии
                    "num_predict": 1000    # Место для развернутого ответа
                }
            })
        return r.json().get("response", "Ошибка LLM")
    except Exception as e: 
        return f"ОШИБКА: Проверь, запущена ли Ollama! ({e})"

if __name__ == "__main__":
    # Проверка папки с корпоративной документацией
    if not os.path.exists("my_notes"): os.makedirs("my_notes")
    
    # Перед индексацией удаляем старые индексы, чтобы избежать каши
    for f in ["index.faiss", "docs.pkl"]:
        if os.path.exists(f): os.remove(f)

    print("⏳ Идет глубокая индексация базы (50 МБ)...")
    chunks = []
    chunk_size = 800 # Оптимальный размер куска для модели E5
    
    for f_name in os.listdir("my_notes"):
        file_path = os.path.join("my_notes", f_name)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            # Нарезка всего файла на части, чтобы ничего не пропустить
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                # Модель E5 требует префикса 'passage: ' для хранимых данных
                chunks.append(f"Файл: {f_name} | Текст: {chunk}")
    
    if chunks:
        # Генерация эмбеддингов на RTX 4060
        embs = model.encode(chunks)
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(np.array(embs).astype("float32"))
        faiss.write_index(index, "index.faiss")
        with open("docs.pkl", "wb") as f: pickle.dump(chunks, f)
        print("✅ БАЗА ПРОИНДЕКСИРОВАНА. СИСТЕМА ГОТОВА.")
    
    while True:
        user_q = input("\n🔎 Запрос: ")
        if user_q.lower() in ['exit', 'выход', 'quit']: break
        
        context = get_context(user_q)
        # Для отладки можно распечатать context, чтобы видеть, что нашел FAISS
        # print(f"DEBUG: Найдено кусков: {len(context)}") 
        
        answer = ask_ollama(user_q, context)
        print("\n✅ ОТВЕТ:\n" + answer)