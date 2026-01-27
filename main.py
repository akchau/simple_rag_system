import requests, faiss, pickle, os, fitz, numpy as np, time
from sentence_transformers import SentenceTransformer

# 1. Глобальные настройки
# Оптимальная модель для 8GB VRAM твоей видеокарты
model = SentenceTransformer("intfloat/multilingual-e5-small")

# Используем только имена файлов для стабильности в Windows
INDEX_FILE = "index.faiss"
DOCS_FILE = "docs.pkl"
NOTES_DIR = "my_notes"

def extract_text(path):
    """Извлечение текста из PDF-кодексов."""
    text = ""
    try:
        with fitz.open(path) as doc:
            for page in doc: text += page.get_text()
    except Exception as e: 
        print(f"❌ Ошибка PDF {path}: {e}")
    return text

def ask_ollama(q, ctx):
    """Запрос к Qwen 2.5 с оптимизированным контекстом."""
    prompt = (
        f"### РОЛЬ: Ты — педантичный российский юрист. Твоя база — ТОЛЬКО предоставленный текст.\n"
        f"### ПРАВИЛА:\n"
        f"1. ЦИТИРУЙ ДОСЛОВНО. Не меняй юридические формулировки.\n"
        f"2. Выделяй **ЖИРНЫМ** ключевые требования (например, **участие специалиста**).\n"
        f"3. Пиши кратко и по существу. Если ответа нет, пиши 'ИНФОРМАЦИЯ НЕ НАЙДЕНА'.\n\n"
        f"### КОНТЕКСТ:\n{ctx}\n\n"
        f"### ВОПРОС: {user_q}\n\n"
        f"### ЮРИДИЧЕСКИЙ ОТВЕТ:"
    )
    
    start_gen = time.time()
    try:
        r = requests.post("http://localhost:11434/api/generate", 
            json={
                "model": "qwen2.5", 
                "prompt": prompt, 
                "stream": False, 
                "options": {
                    "num_ctx": 8192,     # Оптимизация: уменьшили с 16k для скорости
                    "temperature": 0.0, 
                    "num_predict": 500   # Ограничение длины ответа
                }
            })
        ans = r.json().get("response", "Ошибка модели")
        duration = time.time() - start_gen
        return ans, duration
    except Exception as e:
        return f"ОШИБКА: Ollama не отвечает. ({e})", 0

if __name__ == "__main__":
    print(f"📍 РАБОЧАЯ ПАПКА: {os.getcwd()}") # Убедись, что это C:\cyber_win
    
    loaded = False
    # Пытаемся загрузить готовую базу
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        try:
            print("⚡ Загружаю базу...")
            index = faiss.read_index(INDEX_FILE)
            with open(DOCS_FILE, "rb") as f:
                chunks = pickle.load(f)
            print(f"✅ База готова. Фрагментов: {len(chunks)}")
            loaded = True
        except Exception as e:
            print(f"⚠️ Ошибка загрузки (пересоздаю): {e}")

    if not loaded:
        # Первичная индексация
        print("⏳ Начинаю первичную индексацию (около 1.5 мин)...")
        if not os.path.exists(NOTES_DIR) or not os.listdir(NOTES_DIR):
            print(f"❌ ОШИБКА: Положи PDF в папку {NOTES_DIR}!")
            exit()

        chunks = []
        for f_name in os.listdir(NOTES_DIR):
            if f_name.lower().endswith(".pdf"):
                path = os.path.join(NOTES_DIR, f_name)
                raw_text = extract_text(path)
                if raw_text:
                    print(f"📖 Обработка {f_name}...")
                    for i in range(0, len(raw_text), 1100): 
                        chunks.append(f"Файл: {f_name} | passage: {raw_text[i:i+1500]}")

        if chunks:
            embs = model.encode(chunks)
            index = faiss.IndexFlatL2(embs.shape[1])
            index.add(np.array(embs).astype("float32"))
            faiss.write_index(index, INDEX_FILE)
            with open(DOCS_FILE, "wb") as f:
                pickle.dump(chunks, f)
            print(f"🚀 База создана: {len(chunks)} фрагментов.")

    # Цикл запросов
    while True:
        user_q = input("\n🔎 Юридический запрос (или 'выход'): ")
        if user_q.lower() in ['exit', 'выход', 'quit']: break
        
        # 1. Поиск (Оптимизация: k=7 вместо 10)
        start_search = time.time()
        v = model.encode(["query: " + user_q])
        _, ids = index.search(np.array(v).astype("float32"), 7) 
        ctx = "\n---\n".join([chunks[i] for i in ids[0]])
        search_time = time.time() - start_search
        
        # 2. Генерация
        print(f"⏳ Поиск завершен за {search_time:.2f} сек. Генерирую ответ...")
        answer, gen_time = ask_ollama(user_q, ctx)
        
        print("\n✅ ОТВЕТ:\n" + answer)
        print(f"\n📊 Тайминги: Поиск: {search_time:.2f}с | Генерация: {gen_time:.2f}с | Итого: {search_time+gen_time:.2f}с")